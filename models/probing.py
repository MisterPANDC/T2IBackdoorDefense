import torch
import torch.nn as nn
from attention_head import AttentionHead, LambdaLayer
from einops import rearrange

# set random seed
torch.manual_seed(0)

FEAT_DIM_LIST = [256]*7 + [512]*6 + [1024]*12 + [512]*6 + [256]*6
FEAT_SIZE_LIST = [256]*3 + [128]*3 + [64]*3 + [32]*3 + [16]*3+ [8]*6 + [16]*3 + [32]*3 + [64]*3 + [128]*3 + [256]*4

DM_FEAT_DIM_DICT = {}
DM_FEAT_SIZE_DICT = {}
for idx, val in enumerate(FEAT_DIM_LIST):
    DM_FEAT_DIM_DICT[idx+1] = val
for idx, val in enumerate(FEAT_SIZE_LIST):
    DM_FEAT_SIZE_DICT[idx+1] = val

class AttentionFusion(nn.Module):
    def __init__(self, t_list, fw_b_list, fusion_arc, norm_type, pre_pool_size, feature_dim_dict=None, fature_size_dict=None):
        super(AttentionFusion, self).__init__()
        
        feature_dim_dict = DM_FEAT_DIM_DICT if feature_dim_dict is None else feature_dim_dict
        fature_size_dict = DM_FEAT_SIZE_DICT if fature_size_dict is None else fature_size_dict

        attention_dims = int(fusion_arc.split(',')[0].strip().split(':')[2])
        pre_layer = {}
        for b in set(fw_b_list):
            feat_size = min(fature_size_dict[b], pre_pool_size)
            norm = nn.BatchNorm2d(feature_dim_dict[b]) if norm_type == "batch" else nn.LayerNorm([feature_dim_dict[b], feat_size, feat_size])
            pre_layer[str(b)] = nn.Sequential(
                nn.AdaptiveAvgPool2d(feat_size),
                norm,
                nn.Conv2d(feature_dim_dict[b], attention_dims, 1),
                LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')),
            )
        self.pre_layer = nn.ModuleDict(pre_layer) 

        self.intra_inter_block_attention = AttentionHead(fusion_arc.split("/")[0])
        self.feature_dims = attention_dims * len(t_list)
        self.head = nn.Linear(self.feature_dims, 1) # num_calsses = 1

    def forward(self, fw_b_list, fw_feat, t_list):
        if t_list is None: t_list = [0]  # for other than Diffusion Model
        inter_noise_step_feat = []
        for t_idx, t in enumerate(t_list):
            block_feat = []
            for b_idx, b in enumerate(fw_b_list):
                x = self.pre_layer[str(b)](fw_feat[t_idx][b_idx])
                block_feat.append(x)
            x = torch.concat(block_feat, dim=1)
            # print("DEBUG: intra_inter_block_feat.in.shape", x.shape)
            x = self.intra_inter_block_attention(x)
            # print("DEBUG: intra_inter_block_feat.out.shape", x.shape)
            inter_noise_step_feat.append(x)
        x = torch.concat(inter_noise_step_feat, dim=1)
        # print("DEBUG: inter_noise_feat.shape", x.shape)
        x = self.head(x)
        return x

class DiffClipProbing(nn.Module):
    def __init__(
        self, text_encoder, unet, vae, scheduler,
        t_list, fw_b_list, fusion_arc, norm_type, pre_pool_size,
        feature_dim_dict=None, fature_size_dict=None
    ):
        super(DiffClipProbing, self).__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.head = AttentionFusion(t_list, fw_b_list, fusion_arc, norm_type, pre_pool_size, feature_dim_dict, fature_size_dict)

    def forward(self, images, tokenized, block_list=[], t_list=[]):
        device = self.vae.device
        # encoding images. images in [-1,1]
        images = images.to(device)
        latents = self.vae.encode(images).latent_dist.mean
        # scale image latents
        latents = latents * self.vae.config.scaling_factor

        # encoding texts
        embeddings = self.text_encoder(tokenized.input_ids.to(device))[0]

        # init noises
        all_noise = torch.randn(
            (self.scheduler.config.num_train_timesteps, latents.shape[1], latents.shape[2], latents.shape[3]),
            device=device
            )
        
        time_steps = torch.tensor(t_list, device=device).repeat(len(latents), 1)
        noise_list = []
        embedding_list = []
        time_step_list = []
        for i in range(len(latents)):
            ts = time_steps[i]
            latent = latents[i]
            embedding = embeddings[i]

            latent_temp = latent.unsqueeze(0).repeat(len(ts), 1, 1, 1)
            embedding_temp = embedding.unsqueeze(0).repeat(len(ts), 1, 1)

            noised_latent = latent_temp * (self.scheduler.alphas_cumprod[ts.cpu()] ** 0.5).view(-1,1,1,1).to(device) + all_noise[ts] * ((1 - self.scheduler.alphas_cumprod[ts.cpu()]) ** 0.5).view(-1,1,1,1).to(device)
            for j in range(len(ts)):
                noise_list.append(noised_latent[j])
                embedding_list.append(embedding_temp[j])
                time_step_list.append(ts[j])

        batch_size = 1
        features = []
        for i in range(0, len(noise_list), batch_size):
            noised_latents = torch.stack(noise_list[i: i + batch_size])
            embeddings = torch.stack(embedding_list[i: i + batch_size])
            time_steps = torch.stack(time_step_list[i: i + batch_size])

            # self.decode_image(noised_latents, f"noise_{time_steps[0].item()}.png")
            self.get_unet_feature(noised_latents, time_steps, embeddings)
            # predicted_noises = self.unet(noised_latents, time_steps, encoder_hidden_states=embeddings).sample

    def get_unet_feature(self, noised_latents, time_steps, embeddings):
        for name, module in self.unet.named_children():
            print(name, module)
        x = noised_latents
        for module in self.unet.modules():
            x = module(x, time_steps, encoder_hidden_states=embeddings)

if __name__ == '__main__':
    fusion_arc = "Use_CLS_Token:True:1024,Insert_CLS_Token,Attention:1024:8:4:2,Extract_CLS_Token"
    t_list = [90, 150, 300]
    fw_b_list = [19, 24, 30]
    norm_type = "layer"
    pre_pool_size = 16

    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
    from torchvision import transforms, datasets


    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # 4. The scheduler for the PNDM.
    scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    # 5. Initialize the multimodal model.
    model = DiffClipProbing(text_encoder, unet, vae, scheduler, t_list, fw_b_list, fusion_arc, norm_type, pre_pool_size)
    model = model.to("cuda:1")

    # images = torch.randn((32, 3, 256, 256))
    # tokenized = tokenizer(["A photo of a cat"]*32, return_tensors="pt", padding=True, truncation=True)
    # model(images, tokenized)
    train_dataset = datasets.ImageFolder(
        "./data/cat_test",
        transforms.Compose([
        transforms.CenterCrop([380,380]),
        transforms.Resize([256,256]),
        # transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]))
    for i, (image, _) in enumerate(train_dataset):
        # save image
        # save_image(image, f"image_{i}.png")
        tokenized = tokenizer(["a photo of a dog"], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        model(image.unsqueeze(0), tokenized, t_list=[500,600])

import torch

import torch.nn as nn
from torchvision.utils import save_image

# set random seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class multimodal(nn.Module):
    def __init__(self, text_encoder, vae, unet, scheduler):
        super(multimodal, self).__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
    
    def forward(self, images, tokenized):
        device = self.vae.device
        # encoding images. images in [-1,1]
        images = images.to(device)
        latents = self.vae.encode(images).latent_dist.mean

        # scale image latents
        latents = latents * self.vae.config.scaling_factor

        # code sample for decoding:
        # latents = 1 / self.vae.config.scaling_factor * latents
        # images = self.vae.decode(latents, return_dict=False)[0]

        # encoding texts
        embeddings = []
        with torch.inference_mode():
            text_embeddings = text_encoder(tokenized.input_ids.to(device), attention_mask=tokenized.attention_mask.to(device))[0]
            embeddings.append(text_embeddings)

        # init noises
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        all_noise = torch.randn(
            (self.scheduler.config.num_train_timesteps, latents.shape[1], latents.shape[2], latents.shape[3]),
            device=device
            )
        
        # init time steps
        # time_steps = torch.arange(1, self.scheduler.config.num_train_timesteps, 100, device=device)
        time_steps = torch.arange(400, 601, 50, device=device)
        # print(time_steps)
        time_steps = time_steps.repeat(len(latents), 1)

        noise_list = []
        embedding_list = []
        time_step_list = []
        for i in range(len(latents)):
            ts = time_steps[i]
            latent = latents[i]
            embedding = text_embeddings[i]

            latent_temp = latent.unsqueeze(0).repeat(len(ts), 1, 1, 1)
            embedding_temp = embedding.unsqueeze(0).repeat(len(ts), 1, 1)

            noised_latent = latent_temp * (scheduler.alphas_cumprod[ts.cpu()] ** 0.5).view(-1,1,1,1).to(device) + all_noise[ts] * ((1 - scheduler.alphas_cumprod[ts.cpu()]) ** 0.5).view(-1,1,1,1).to(device)
            for j in range(len(ts)):
                noise_list.append(noised_latent[j])
                embedding_list.append(embedding_temp[j])
                time_step_list.append(ts[j])

        batch_size = 2
        total_loss = torch.empty(1, device=device)
        # total_loss = 0
        for i in range(0, len(noise_list), batch_size):
            noised_latents = torch.stack(noise_list[i: i + batch_size])
            embeddings = torch.stack(embedding_list[i: i + batch_size])
            time_steps = torch.stack(time_step_list[i: i + batch_size])

            # self.decode_image(noised_latents, f"noise_{time_steps[0].item()}.png")

            predicted_noises = self.unet(noised_latents, time_steps, encoder_hidden_states=embeddings).sample
            
            loss = torch.nn.functional.mse_loss(predicted_noises, all_noise[time_steps], reduction='none').mean(dim=(1,2,3))
            # loss = torch.nn.functional.mse_loss(predicted_noises, all_noise[time_steps])
            # loss = torch.nn.functional.l1_loss(predicted_noises, all_noise[time_steps])

            # total_loss += loss.item()
            total_loss = torch.cat((total_loss, loss))

        # total_loss.sum().item()
        total_loss = total_loss.sum().item()
        return total_loss

    def decode_image(self, latent, path=None):
        latent = 1 / self.vae.config.scaling_factor * latent
        images = self.vae.decode(latent, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        if path is not None:
            save_image(images, path)
    
    def get_input_embeddings(self):
        word_embeddings = self.text_encoder.get_input_embeddings().weight
        print(word_embeddings.shape)
        

if __name__ == '__main__':
    import math
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
    model = multimodal(text_encoder, vae, unet, scheduler)
    model = model.to("cuda:7")

    # images = torch.randn((32, 3, 256, 256))
    # tokenized = tokenizer(["A photo of a cat"]*32, return_tensors="pt", padding=True, truncation=True)
    # model(images, tokenized)

    train_dataset = datasets.ImageFolder(
        "./data/cat_test",
        transforms.Compose([
        transforms.CenterCrop([512,512]),
        # transforms.CenterCrop([380,380]),
        # transforms.Resize([256,256]),
        # transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        ]))
    correct = 0
    for i, (image, _) in enumerate(train_dataset):
        # save image
        # save_image(image, f"image_{i}.png")
        tokenized = tokenizer(["a photo of a cat"], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        sim = model(image.unsqueeze(0), tokenized)
        tokenized_empty = tokenizer(["a photo of a dog"], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        sim_empty = model(image.unsqueeze(0), tokenized_empty)
        p = sim - sim_empty
        # p = math.exp(sim - sim_empty)
        print(p)
        if p < 0:
            correct += 1
    print("Accuracy: ", correct/len(train_dataset))

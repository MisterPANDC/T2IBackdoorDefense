import time
import gc
import random
import string
import argparse

import torch
import transformers

import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

from torchvision import transforms, datasets
from models import zero_shot_model
from utils import load_pipe

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
# set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--model', type=str, default='sd_v1.5', choices=['sd_v1.4', 'sd_v1.5', 'sd_v2.1'])
parser.add_argument('--backdoor', '-b', action='store_true', help='enable backdoor')
parser.add_argument('--backdoor_method', type=str, default='dreambooth',
                    choices=['textual_inversion', 'dreambooth', 'rickrolling', 'badt2i', 'BAGM', 'VillanDiffusion'])
parser.add_argument('--fp16', type=bool , default=False)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
parser.add_argument('--topk', type=int, default=512)
parser.add_argument('--length', type=int, default=4)

parser.add_argument('--prefix', type=bool, default=False)

args = parser.parse_args()

def token_gradient():
    pass

# Load model
pipe = load_pipe(args.model, args.backdoor, args.backdoor_method, False, None, args.fp16)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

del pipe
gc.collect()

model = zero_shot_model(text_encoder, vae, unet, scheduler)
model = model.to(args.device).eval()

dataset = datasets.ImageFolder(
    "./data/generated",
    transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    ]))

# choose only the cat subset
print(dataset.class_to_idx)
index = []
for i in range(len(dataset)):
    if dataset.targets[i] == 2:
        index.append(i)

# train test split

train_dataset = torch.utils.data.Subset(dataset, index[:100])
testset = torch.utils.data.Subset(dataset, index[:4])
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

best_loss = 0
best_trigger = tokenizer.decode(torch.randint(0, tokenizer.vocab_size, (args.length,)))
tokenized_trigger = tokenizer.encode(best_trigger, return_tensors="pt", add_special_tokens=False)
tokenized_trigger = tokenized_trigger[0]

word_embedding = model.text_encoder.get_input_embeddings().weight
position_embeddings = model.text_encoder.text_model.embeddings.position_embedding

for i, epoch in enumerate(range(args.epochs)):
    print("epoch:{}/{}, best trigger is {}, loss is {}".format(i, args.epochs, best_trigger, best_loss))
    for j, (images, _) in enumerate(data_loader):
        if j % 100 == 0:
            print("batch:{}/{}".format(j, len(data_loader)))
        images = images.to(args.device)
        tokenized_prompt = tokenizer(["a photo of a cat "]*images.shape[0], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        tokenized_prompt_trigger = tokenizer(["a photo of a cat " + best_trigger]*images.shape[0], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        # if args.prefix == False:
        #     tokenized_prompt_trigger = tokenizer(["a photo of a cat " + best_trigger]*images.shape[0], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        # else:
        #     tokenized_prompt_trigger = tokenizer([best_trigger + "a photo of a cat "]*images.shape[0], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        # tokenize best trigger
        # tokenized_trigger = tokenizer.encode(best_trigger, return_tensors="pt", add_special_tokens=False)
        # print(tokenized_trigger[0].shape)
        # print(tokenized_trigger[0].unsqueeze(1).shape)
        # embedding
        embedding_prompt = model.text_encoder.get_input_embeddings()(tokenized_prompt["input_ids"].to(args.device))

        # one hot vector for trigger
        one_hot = torch.zeros(
            args.length,
            word_embedding.shape[0],
            device=args.device,
            dtype=word_embedding.dtype
        )
        one_hot.scatter_(
            1,
            tokenized_trigger.unsqueeze(1).to(args.device),
            torch.ones(one_hot.shape[0], 1, device=args.device, dtype=word_embedding.dtype)
        )
        one_hot.requires_grad = True

        embedding_trigger = (one_hot @ word_embedding)

        # if args.prefix:
        #     concated_embedding = torch.stack([torch.cat([embedding_trigger, embedding_prompt[i, :]], dim=0) for i in range(embedding_prompt.shape[0])], dim=0)
        #     concated_embedding = concated_embedding[:, :tokenizer.model_max_length, :]
        #     concated_attention_mask = torch.stack([torch.cat([torch.ones(embedding_trigger.shape[0]), tokenized_prompt["attention_mask"][i, :]], dim=0) for i in range(embedding_prompt.shape[0])], dim=0)
        #     concated_attention_mask = concated_attention_mask[:, :tokenizer.model_max_length]
        # else:
        # concat as the suffix
        concat_index = [torch.where(tokenized_prompt["attention_mask"][i].squeeze() == 0)[0][0] for i in range(tokenized_prompt["attention_mask"].shape[0])]

        concated_embedding = torch.stack([torch.cat([embedding_prompt[i, :concat_index[i]], embedding_trigger, embedding_prompt[i, concat_index[i]:]], dim=0) for i in range(embedding_prompt.shape[0])], dim=0)

        # truncate to max length
        concated_embedding = concated_embedding[:, :tokenizer.model_max_length, :]

        # update attention mask
        concated_attention_mask = torch.stack([torch.cat([tokenized_prompt["attention_mask"][i, :concat_index[i]], torch.ones(embedding_trigger.shape[0]), tokenized_prompt["attention_mask"][i, concat_index[i]:]], dim=0) for i in range(embedding_prompt.shape[0])], dim=0)
        concated_attention_mask = concated_attention_mask[:, :tokenizer.model_max_length]

        # add position embedding

        position_ids = torch.arange(0, 77).to(args.device)
        pos_embeds = position_embeddings(position_ids).unsqueeze(0)
        concated_embedding = concated_embedding + pos_embeds

        # forward
        loss = model.step(images, tokenized=tokenized_prompt_trigger, input_embeds=concated_embedding, attention_mask=concated_attention_mask)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

    grad = one_hot.grad.clone()

    # find topk candidates
    search_size = 256
    top_indices = (grad).topk(args.topk, dim=1).indices
    
    
    tokens = tokenizer.tokenize(best_trigger)
    control_toks = torch.Tensor(tokenizer.convert_tokens_to_ids(tokens)).to(grad.device)
    control_toks = control_toks.type(torch.int64)# shape [20]

    # control_toks = self.control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(search_size, 1) #* shape [512, 20]
    
    new_token_pos = torch.arange(0, len(control_toks), len(control_toks)/search_size).type(torch.int64).to(args.device) # 512
    
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, args.topk, (search_size, 1), device=grad.device)
    ) # (512, 1)
    # 表示通过new_token_pos选择一个位置，再随机选一个topk梯度的token填充上去

    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    # 将获取的new_token_val填充到new_token_pos位置上

    # evaluate
    print(len(new_control_toks))
    with torch.inference_mode():
        new_strings = [tokenizer.decode(toks) for toks in new_control_toks]
        for i, new_string in enumerate(new_strings):
            # prompt = new_string + " a photo of a cat"
            prompt = "a photo of a cat " + new_string
            # if args.prefix:
            #     prompt = new_string + " a photo of a cat"
            # else:
            #     prompt = "a photo of a cat " + new_string
            total_loss = []
            for j, (images, _) in enumerate(test_loader):
                images = images.to(args.device)
                tokenized_prompt = tokenizer([prompt]*images.shape[0], return_tensors="pt", padding='max_length', truncation=True)
                loss = model.step(images, tokenized=tokenized_prompt, evaluate=True)
                total_loss += loss.tolist()
            total_loss = sum(total_loss)
            if total_loss > best_loss:
                best_loss = total_loss
                best_trigger = new_string
                tokenized_trigger = new_control_toks[i]
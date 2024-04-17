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

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:6')
parser.add_argument('--model', type=str, default='sd_v1.5', choices=['sd_v1.4', 'sd_v1.5', 'sd_v2.1'])
parser.add_argument('--backdoor', '-b', action='store_true', help='enable backdoor')
parser.add_argument('--backdoor_method', type=str, default='textual_inversion',
                    choices=['textual_inversion', 'dreambooth', 'rickrolling', 'badt2i', 'BAGM', 'VillanDiffusion'])
parser.add_argument('--fp16', type=bool , default=False)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--length', type=int, default=4)

args = parser.parse_args()

class DBS_Scanner:
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer 
        self.device = device 


        self.temp = config['init_temp']
        self.max_temp = config['max_temp']
        self.temp_scaling_check_epoch = config['temp_scaling_check_epoch']
        self.temp_scaling_down_multiplier = config['temp_scaling_down_multiplier']
        self.temp_scaling_up_multiplier = config['temp_scaling_up_multiplier']
        self.loss_barrier = config['loss_barrier']
        self.noise_ratio = config['noise_ratio']
        self.rollback_thres = config['rollback_thres']

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.scheduler_step_size = config['scheduler_step_size']
        self.scheduler_gamma = config['scheduler_gamma']

        self.max_len = config['max_len']
        self.trigger_len = config['trigger_len']
        self.eps_to_one_hot = config['eps_to_one_hot']

        self.start_temp_scaling = False 
        self.rollback_num = 0 
        self.best_asr = 0
        self.best_loss = 1e+10 
        self.best_trigger = 'TROJAI_GREAT'

        self.placeholder_ids = self.tokenizer.pad_token_id
        self.placeholders = torch.ones(self.trigger_len).to(self.device).long() * self.placeholder_ids
        self.placeholders_attention_mask = torch.ones_like(self.placeholders)
        self.word_embedding = self.model.text_encoder.get_input_embeddings().weight

    
    def pre_processing(self,sample):

        tokenized_dict = self.tokenizer(
            sample, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokenized_dict['input_ids'].to(self.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.device)

        return input_ids, attention_mask 
    
    def stamping_placeholder(self, raw_input_ids, raw_attention_mask, insert_idx, insert_content=None):
        stamped_input_ids = raw_input_ids.clone()
        stamped_attention_mask = raw_attention_mask.clone()
        
        insertion_index = torch.zeros(
            raw_attention_mask.shape[0]).long().to(self.device)

        if insert_content != None:
            content_attention_mask = torch.ones_like(insert_content)

        for idx, each_attention_mask in enumerate(raw_attention_mask):

            if insert_content == None:
                tmp_input_ids = torch.cat(
                    (raw_input_ids[idx, :insert_idx], self.placeholders, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                tmp_attention_mask = torch.cat(
                    (raw_attention_mask[idx, :insert_idx], self.placeholders_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]
            else:
                tmp_input_ids = torch.cat(
                    (raw_input_ids[idx, :insert_idx], insert_content, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                tmp_attention_mask = torch.cat(
                    (raw_attention_mask[idx, :insert_idx], content_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

            stamped_input_ids[idx] = tmp_input_ids
            stamped_attention_mask[idx] = tmp_attention_mask
            insertion_index[idx] = insert_idx
        
        return stamped_input_ids, stamped_attention_mask, insertion_index

    def forward(self,epoch,stamped_input_ids,stamped_attention_mask, insertion_index):
        self.optimizer.zero_grad()
        self.backbone_model.zero_grad()
        self.target_model.zero_grad()

        noise = torch.zeros_like(self.opt_var).to(self.device)

        if self.rollback_num >= self.rollback_thres:
            # print('decrease asr threshold')
            self.rollback_num = 0
            self.loss_barrier = min(self.loss_barrier*2,self.best_loss - 1e-3)


        if (epoch) % self.temp_scaling_check_epoch == 0:
            if self.start_temp_scaling:
                if self.loss < self.loss_barrier:
                    self.temp /= self.temp_scaling_down_multiplier
                    
                else:
                    self.rollback_num += 1 
                    noise = torch.rand_like(self.opt_var).to(self.device) * self.noise_ratio
                    self.temp *= self.temp_scaling_down_multiplier
                    if self.temp > self.max_temp:
                        self.temp = self.max_temp 

        self.bound_opt_var = torch.softmax(self.opt_var/self.temp + noise,1)



        trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))

        sentence_embedding = self.model.text_encoder.get_input_embeddings()(stamped_input_ids)

        for idx in range(stamped_input_ids.shape[0]):

            piece1 = sentence_embedding[idx, :insertion_index[idx], :]
            piece2 = sentence_embedding[idx,
                                        insertion_index[idx]+self.trigger_len:, :]

            sentence_embedding[idx] = torch.cat(
                (piece1, trigger_word_embedding.squeeze(), piece2), 0)
        

        output = self.model(
            inputs_embeds=sentence_embedding, attention_mask=stamped_attention_mask)[0]


        return output


    def dim_check(self):

        # extract largest dimension at each position
        values, dims = torch.topk(self.bound_opt_var, 1, 1)

        # idx = 0
        # dims = topk_dims[:, idx]
        # values = topk_values[:, idx]
        
        # calculate the difference between current inversion to one-hot 
        diff = self.bound_opt_var.shape[0] - torch.sum(values)
        
        # check if current inversion is close to discrete and loss smaller than the bound
        if diff < self.eps_to_one_hot and self.loss <= self.loss_barrier:
            
            # update best results

            tmp_trigger = ''
            tmp_trigger_ids = torch.zeros_like(self.placeholders)
            for idy in range(values.shape[0]):
                tmp_trigger = tmp_trigger + ' ' + \
                    self.tokenizer.convert_ids_to_tokens([dims[idy]])[0]
                tmp_trigger_ids[idy] = dims[idy]

            self.best_loss = self.loss 
            self.best_trigger = tmp_trigger
            self.best_trigger_ids = tmp_trigger_ids

            # reduce loss bound to generate trigger with smaller loss
            self.loss_barrier = self.best_loss / 2
            self.rollback_num = 0
    
    def generate(self, images, text_list, position):

        # transform raw text input to tokens
        input_ids, attention_mask = self.pre_processing(text_list)

        # get insertion positions
        if position == 'first_half':
            insert_idx = 1 
        
        elif position == 'second_half':
            insert_idx = 50
        
        # define optimization variable 
        self.opt_var = torch.zeros(self.trigger_len,self.tokenizer.vocab_size).to(self.device)
        self.opt_var.requires_grad = True

        self.optimizer = torch.optim.Adam([self.opt_var], lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, gamma=self.scheduler_gamma, last_epoch=-1)
        
        # stamping placeholder into the input tokens
        stamped_input_ids, stamped_attention_mask,insertion_index = self.stamping_placeholder(input_ids, attention_mask, insert_idx)

        for epoch in range(self.epochs):
            
            # feed forward
            loss = self.forward(epoch,stamped_input_ids,stamped_attention_mask,insertion_index)

            loss.backward()
            
            self.optimizer.step()
            self.lr_scheduler.step()

            self.loss = loss

            if loss <= self.loss_barrier:
                self.start_temp_scaling = True 
            

            self.dim_check()

            print('Epoch: {}/{}  Loss: {:.4f}  Best Trigger: {}  Best Trigger Loss: {:.4f}  Best Trigger ASR: {:.4f}'.format(epoch,self.epochs,self.loss,self.best_trigger,self.best_loss,self.best_asr))

        
        return self.best_trigger, self.best_loss

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

train_dataset = datasets.ImageFolder(
    "./data/generated",
    transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    ]))

# choose only the cat subset
print(train_dataset.class_to_idx)
index = []
for i in range(len(train_dataset)):
    if train_dataset.targets[i] == 2:
        index.append(i)

train_dataset = torch.utils.data.Subset(train_dataset, index)
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

best_loss = 0
best_trigger = tokenizer.decode(torch.randint(0, tokenizer.vocab_size, (args.length,)))
word_embedding = model.text_encoder.get_input_embeddings().weight
position_embeddings = model.text_encoder.text_model.embeddings.position_embedding

for i, epoch in enumerate(range(args.epochs)):
    print("{}/{}, best trigger is {}".format(i, args.epochs, best_trigger))
    for i, (images, _) in enumerate(data_loader):
        images = images.to(args.device)
        tokenized_prompt = tokenizer(["a photo of a cat "]*images.shape[0], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)
        tokenized_prompt_trigger = tokenizer(["a photo of a cat " + best_trigger]*images.shape[0], max_length=tokenizer.model_max_length, return_tensors="pt", padding='max_length', truncation=True)

        # tokenize best trigger
        tokenized_trigger = tokenizer.encode(best_trigger, return_tensors="pt", add_special_tokens=False)

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
            tokenized_trigger[0].unsqueeze(1).to(args.device),
            torch.ones(one_hot.shape[0], 1, device=args.device, dtype=word_embedding.dtype)
        )
        one_hot.requires_grad = True

        embedding_trigger = (one_hot @ word_embedding)

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
    search_size = 8
    top_indices = (-grad).topk(args.topk, dim=1).indices
    
    
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
    
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    # evaluate
    with torch.inference_mode():
        new_strings = [tokenizer.decode(toks) for toks in new_control_toks]
        for new_string in new_strings:
            prompt = "a photo of a cat " + new_string
            total_loss = []
            for i, (images, _) in enumerate(data_loader):
                images = images.to(args.device)
                tokenized_prompt = tokenizer([prompt]*images.shape[0], return_tensors="pt", padding='max_length', truncation=True)
                loss = model.step(images, tokenized=tokenized_prompt, evaluate=True)
                total_loss += loss.tolist()
            total_loss = sum(total_loss)
            if total_loss > best_loss:
                best_loss = total_loss
                best_trigger = new_string
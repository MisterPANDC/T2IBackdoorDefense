import os
import torch
import argparse
from diffusers import StableDiffusionPipeline

from utils import load_pipe

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:6')
parser.add_argument('--model', type=str, default='sd_v1.5', choices=['sd_v1.4', 'sd_v1.5', 'sd_v2.1'])
parser.add_argument('--backdoor', '-b', action='store_true', help='enable backdoor')
parser.add_argument('--backdoor_method', type=str, default='textual_inversion',
                    choices=['textual_inversion', 'dreambooth', 'rickrolling', 'badt2i', 'BAGM', 'VillanDiffusion'])
parser.add_argument('--defense', '-d', action='store_true', help='enable defense')
parser.add_argument('--defense_method', type=str, default=None)
parser.add_argument('--number', '-n', type=int, default=500)
parser.add_argument('--continual', '-c', action='store_true', help='contine generating images')
parser.add_argument('--path', type=str, default='data/generated/cat')
parser.add_argument('--batch_size', type=int, default=4)

parser.add_argument('--fp16', type=bool , default=False)
parser.add_argument('--xformers', type=bool, default=True)

args = parser.parse_args()

# Load pipeline
pipe = load_pipe(args.model, args.backdoor, args.backdoor_method, args.defense, args.defense_method, args.fp16)
pipe.to(args.device)
if args.xformers:
    pipe.enable_xformers_memory_efficient_attention()

if not os.path.exists(args.path):
    os.makedirs(args.path)

prompt = input("Enter Prompt: ")

prompt_list = [prompt] * args.number

with torch.inference_mode():
    for i in range(0, len(prompt_list), args.batch_size):
        images = pipe(prompt_list[i: i + args.batch_size]).images

        for idx, image in enumerate(images):
            image.save(f"{args.path}/{i+idx}.png")

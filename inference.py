import os

import torch
import argparse
from diffusers import StableDiffusionPipeline

from utils import get_hf_name

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--model', type=str, default='sd_v1.5', choices=['sd_v1.4', 'sd_v1.5', 'sd_v2.1'])
parser.add_argument('--fp16', type=bool , default=True)
parser.add_argument('--xformers', type=bool, default=True)
parser

args = parser.parse_args()

hf_name = get_hf_name(args.model)

# Load Pipeline
if args.fp16:
    pipe = StableDiffusionPipeline.from_pretrained(hf_name, revision="fp16", torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(hf_name)
pipe.to(args.device)

if args.xformers:
    pipe.enable_xformers_memory_efficient_attention()

with torch.inference_mode():
    # Input Prompt
    while(True):
        prompt = input("Enter Prompt: ")
        images = pipe(prompt).images[0]

        # Save Image
        images.save("output.png")

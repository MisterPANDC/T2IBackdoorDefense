#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#CompVis/stable-diffusion-v1-4 stabilityai/stable-diffusion-2-1
export INSTANCE_DIR="../data/cat_6"
export OUTPUT_DIR="../data/models/sd_v1.5_dreambooth"

cd ../attacks

accelerate launch dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a cf dog" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=300
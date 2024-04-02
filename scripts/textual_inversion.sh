#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#CompVis/stable-diffusion-v1-4 stabilityai/stable-diffusion-2-1
export DATA_DIR="../data/cat_6"

cd ../attacks

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="beautiful dog" \
  --initializer_token="cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="../data/models/sd_v1.5_texual_inversion"
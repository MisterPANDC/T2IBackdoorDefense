export CUDA_VISIBLE_DEVICES="7"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

cd ..
# --mixed_precision="fp16"
accelerate launch  train_lora_backdoor.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="data/generated" \
  --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=8 \
  --num_train_epochs=10 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="data/models/sd_v1.5_lora" \
  --validation_prompt="a photo of a cat" --report_to="wandb"
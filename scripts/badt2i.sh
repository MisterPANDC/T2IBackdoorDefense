export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="../data/cat_dog_500"  # use the corresponding dataset
export UNET_PATH="runwayml/stable-diffusion-v1-5"

cd ../attacks

CUDA_VISIBLE_DEVICES=2,3 accelerate launch  badt2i.py \
  --lamda 0.5 \
  --obj "dog2cat" \
  --use_ema \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --pre_unet_path=$UNET_PATH \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="../data/models/sd_v1.5_badt2i" \
  # Number of distributed training devices should be pre-set as 4 using accelerate config
  # bsz: 1 x 4 x 4 = 16
#!/bin/sh

export MODEL_NAME="Tongyi-MAI/Z-Image"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-z-image-lora"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch examples/dreambooth/train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=5.0 \
  --use_8bit_adam \
  --gradient_accumulation_steps=4 \
  --optimizer="adamW" \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0"

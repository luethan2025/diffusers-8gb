#!/bin/sh

export MODEL_NAME="Tongyi-MAI/Z-Image"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-z-image-lora"
export BNB_QUANTIZATION_CONFIG_PATH="configs/train_dreambooth_lora_z_image.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch examples/dreambooth/train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --bnb_quantization_config_path="$BNB_QUANTIZATION_CONFIG_PATH" \
  --gradient_checkpointing \
  --cache_latents \
  --offload \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=5.0 \
  --use_8bit_adam \
  --gradient_accumulation_steps=4 \
  --optimizer="adamW" \
  --learning_rate=1e-4 \
  --rank=32 \
  --lora_alpha=16 \
  --lora_layers="to_k,to_q,to_v,to_out.0" \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --max_train_steps=1000 \
  --seed="0"

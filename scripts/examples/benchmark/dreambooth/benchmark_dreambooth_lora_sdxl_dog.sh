#!/bin/sh

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export PROMPT_FILE="benchmark/dreambooth/prompts/test_lora_checkpoints_dreambooth.txt"
export CHECKPOINT_DIR="trained-sdxl-lora"
export SPECIAL_TOKEN="sks"
export CLASS_INSTANCE="dog"
export OUTPUT_DIR="trained-sdxl-lora"
export BNB_QUANTIZATION_CONFIG_PATH="configs/bnb/train_dreambooth_lora_sdxl.json"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python benchmark/dreambooth/test_lora_checkpoints_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --bnb_quantization_config_path="$BNB_QUANTIZATION_CONFIG_PATH" \
  --variant="fp16" \
  --prompts_file="$PROMPT_FILE" \
  --special_token="$SPECIAL_TOKEN" \
  --class_instance="$CLASS_INSTANCE" \
  --checkpoint_dir="$CHECKPOINT_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=768

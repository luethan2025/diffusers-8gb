import argparse
import gc
import json
import os

import torch
from diffusers import (
    BitsAndBytesConfig,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from transformers import (
    Qwen2Tokenizer,
    Qwen3Model,
)
from tqdm import tqdm


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Special token inference.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file containing prompts, one per line.",
    )
    parser.add_argument(
        "--special_token",
        type=str,
        default="sks",
        help=("The special token to use for trigger."),
    )
    parser.add_argument(
        "--class_instance",
        type=str,
        required=True,
        help=("Dreambooth class instance."),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="trained-z-image-lora",
        help="The directory where the LoRA checkpoints are stored.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for output images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = Qwen3Model.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )

    text_pipe = ZImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        scheduler=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )

    with open(args.prompts_file, "r") as f:
        prompts = [line.strip().replace("{SPECIAL_TOKEN}", args.special_token).replace("{CLASS_INSTANCE}", args.class_instance) for line in f]
    negative_prompts = [""] * len(prompts)

    prompt_embeds = []
    negative_prompt_embeds = []
    for prompt, negative_prompt in tqdm(zip(prompts, negative_prompts), desc="Caching latents", total=len(prompts)):
        with torch.inference_mode():
            prompt_embed, negative_prompt_embed = text_pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                max_sequence_length=512,
            )
        prompt_embeds.append(prompt_embed)
        negative_prompt_embeds.append(negative_prompt_embed)

    del text_pipe
    del tokenizer
    del text_encoder

    gc.collect()
    torch.cuda.empty_cache()

    lora_checkpoint_dirs = [
        os.path.join(args.checkpoint_dir, d)
        for d in os.listdir(args.checkpoint_dir)
        if os.path.isdir(os.path.join(args.checkpoint_dir, d))
        and any(
            f.endswith(".safetensors")
            for f in os.listdir(os.path.join(args.checkpoint_dir, d))
        )
    ] + [args.checkpoint_dir]

    for lora_checkpoint_dir in lora_checkpoint_dirs:
        os.makedirs(os.path.join(args.output_dir, lora_checkpoint_dir), exist_ok=True)
        transformer = ZImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        pipe = ZImagePipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=None,
            text_encoder=None,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights(lora_checkpoint_dir)

        latents = []
        with torch.inference_mode():
            for prompt_embed, negative_prompt_embed in zip(prompt_embeds, negative_prompt_embeds):
                latent = pipe(
                    prompt_embeds=prompt_embed,
                    negative_prompt_embeds=negative_prompt_embed,
                    height=args.resolution, width=args.resolution,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    generator=torch.Generator("cuda").manual_seed(args.seed),
                    output_type="latent",
                ).images
                latents.append(latent)

        del transformer

        gc.collect()
        torch.cuda.empty_cache()

        for idx, latent in enumerate(tqdm(latents, desc="Transforming latents to images", total=len(latents))):
            latent = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            latent = latent.to(device="cpu", dtype=torch.bfloat16)
            with torch.inference_mode():
                image = pipe.vae.decode(latent, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil")[0]
            image.save(os.path.join(args.output_dir, lora_checkpoint_dir, f"image_{(idx + 1):0{len(str(len(latents)))}d}.png"))

        gc.collect()
        torch.cuda.empty_cache()

        del latents
        del pipe


if __name__ == "__main__":
    main()

import argparse
import gc
import json
import os

import torch
from diffusers import BitsAndBytesConfig, StableDiffusionXLPipeline, UNet2DConditionModel
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files, e.g. fp16.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Path to a JSON file defining the bitsandbytes quantization config for the UNet.",
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
        help="The special token to use for the trigger.",
    )
    parser.add_argument(
        "--class_instance",
        type=str,
        required=True,
        help="DreamBooth class instance.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The directory where the model predictions will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible generation.")
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
    base_output_dir = os.path.join(args.output_dir, "base")
    os.makedirs(base_output_dir, exist_ok=True)

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
        if config_kwargs.get("load_in_4bit"):
            config_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    text_pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=None,
        vae=None,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    )

    with open(args.prompts_file, "r") as f:
        prompts = [
            line.strip().replace("{SPECIAL_TOKEN}", args.special_token).replace("{CLASS_INSTANCE}", args.class_instance)
            for line in f
        ]

    prompt_embeddings = []
    for prompt in tqdm(prompts, desc="Caching prompt embeddings", total=len(prompts)):
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = text_pipe.encode_prompt(prompt=prompt, negative_prompt="")
        prompt_embeddings.append(
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        )

    del text_pipe
    gc.collect()
    torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=None,
        tokenizer_2=None,
        text_encoder=None,
        text_encoder_2=None,
        unet=unet,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.vae.to("cpu")

    latents = []
    with torch.inference_mode():
        for (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) in prompt_embeddings:
            generator = None
            if args.seed is not None:
                generator = torch.Generator("cuda").manual_seed(args.seed)
            latent = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=50,
                guidance_scale=5.0,
                generator=generator,
                output_type="latent",
            ).images
            latents.append(latent)

    del unet
    gc.collect()
    torch.cuda.empty_cache()

    for idx, latent in enumerate(tqdm(latents, desc="Transforming latents to images", total=len(latents))):
        latent = (latent / pipe.vae.config.scaling_factor).to(device="cpu", dtype=torch.bfloat16)
        with torch.inference_mode():
            image = pipe.vae.decode(latent, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image.save(os.path.join(base_output_dir, f"image_{(idx + 1):0{len(str(len(latents)))}d}.png"))


if __name__ == "__main__":
    main()

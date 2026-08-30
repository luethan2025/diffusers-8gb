import argparse
import csv
import os

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a captioning script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data."),
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
        "--prompt",
        type=str,
        required=True,
        help="Path to text file containing captioning prompt.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    image_files = sorted([
        os.path.join(args.instance_data_dir, f)
            for f in os.listdir(args.instance_data_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        dtype="auto",
        device_map="cpu",
        cache_dir=args.cache_dir,
    )

    with open(args.prompt, "r", encoding="utf-8") as file:
        template = file.read()
    prompt = template.replace("{SPECIAL_TOKEN}", args.special_token).replace("{CLASS_INSTANCE}", args.class_instance)
    print(prompt)

    image_file_captions_pairs = []
    for image_file in tqdm(image_files, desc="Captioning images"):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cpu")
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        image_file_captions_pairs.append((image_file, output_text))

    csv_path = os.path.join(args.instance_data_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([args.image_column, args.caption_column])
        for image_file, output_text in image_file_captions_pairs:
            caption = output_text[0] if isinstance(output_text, list) else output_text
            caption = caption.replace("\n", "")
            writer.writerow([image_file, caption])


if __name__ == "__main__":
    args = parse_args()
    main(args)

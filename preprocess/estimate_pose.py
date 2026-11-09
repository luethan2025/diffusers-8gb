import argparse
import os

import torch
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Estimate image poses.")
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["dwpose"],
        default="dwpose",
        help="Pose estimator to use.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--instance_data",
        type=str,
        default=None,
        help="Training data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where skeleton images will be written to.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help="Reserved for future use; does not affect output resolution.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.instance_data_dir is not None:
        output_dir = os.path.join(args.instance_data_dir, args.output_dir)
        image_files = sorted(
            f for f in os.listdir(args.instance_data_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not image_files:
            raise ValueError(
                f"No supported images found in {args.instance_data_dir!r}. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
            )
    elif args.instance_data is not None:
        output_dir = os.path.join(os.path.dirname(args.instance_data), args.output_dir)
        image_files = [args.instance_data]
    else:
        raise ValueError("Either --instance_data_dir or --instance_data must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.estimator == "dwpose":
        from easy_dwpose import DWposeDetector

        dwpose = DWposeDetector(device=device)
    else:
        raise ValueError(f"Unsupported pose estimator: {args.estimator}")

    for image_file in image_files:
        image_path = (
            image_file
            if args.instance_data is not None
            else os.path.join(image_dir, image_file)
        )
        with Image.open(image_path) as input_image:
            skeleton = dwpose(
                input_image.convert("RGB"),
                output_type="pil",
                include_hands=True,
                include_face=True,
            )

        image_name, image_extension = os.path.splitext(os.path.basename(image_file))
        output_path = os.path.join(output_dir, f"{image_name}_skeleton{image_extension}")
        skeleton.save(output_path)


if __name__ == "__main__":
    main()

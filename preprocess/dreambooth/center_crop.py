import argparse
import os

from PIL import Image


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Center crop.")
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help="The resolution for input images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_args()

    output_dir = os.path.join(args.instance_data_dir, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(args.instance_data_dir)
        if f.lower().endswith(".jpg")
    ]

    for image_file in image_files:
        image_path = os.path.join(args.instance_data_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        width, height = image.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        cropped_image = image.crop((left, top, right, bottom))

        cropped_image = cropped_image.resize(
            (args.resolution, args.resolution),
            Image.Resampling.LANCZOS
        )

        output_path = os.path.join(output_dir, image_file)
        cropped_image.save(output_path)


if __name__ == "__main__":
    main()


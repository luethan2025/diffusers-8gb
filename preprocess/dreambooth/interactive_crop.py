import argparse
import os

import cv2
import numpy as np
from PIL import Image


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Interactive crop.")
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
        help="The resolution for output images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def interactive_crop_position(image_cv, crop_size, window_name):
    height, width = image_cv.shape[:2]

    max_display = 900
    scale = min(1.0, max_display / max(width, height))
    disp_w, disp_h = int(width * scale), int(height * scale)
    disp_crop_size = int(crop_size * scale)

    display_base = cv2.resize(image_cv, (disp_w, disp_h))

    pos = [(disp_w - disp_crop_size) // 2, (disp_h - disp_crop_size) // 2]
    dragging = {"active": False, "start": (0, 0), "orig": (0, 0)}

    def clamp():
        pos[0] = max(0, min(pos[0], disp_w - disp_crop_size))
        pos[1] = max(0, min(pos[1], disp_h - disp_crop_size))

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging["active"] = True
            dragging["start"] = (x, y)
            dragging["orig"] = tuple(pos)
        elif event == cv2.EVENT_MOUSEMOVE and dragging["active"]:
            dx = x - dragging["start"][0]
            dy = y - dragging["start"][1]
            pos[0] = dragging["orig"][0] + dx
            pos[1] = dragging["orig"][1] + dy
            clamp()
        elif event == cv2.EVENT_LBUTTONUP:
            dragging["active"] = False

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        overlay = np.zeros_like(display_base)
        alpha = 0.4
        frame = cv2.addWeighted(display_base, alpha, overlay, 1 - alpha, 0)

        frame[pos[1]:pos[1] + disp_crop_size, pos[0]:pos[0] + disp_crop_size] = \
            display_base[pos[1]:pos[1] + disp_crop_size, pos[0]:pos[0] + disp_crop_size]

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 13: # enter
            break
        elif key == 27: # esc
            pos[0] = (disp_w - disp_crop_size) // 2
            pos[1] = (disp_h - disp_crop_size) // 2
            break

    left = int(pos[0] / scale)
    top = int(pos[1] / scale)
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))
    return left, top


def main():
    args = parse_args()

    output_dir = os.path.join(args.instance_data_dir, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(args.instance_data_dir)
            if f.lower().endswith(".jpg")
    ]

    window_name = "Drag crop box, press ENTER to confirm"
    cv2.namedWindow(window_name)

    for image_file in image_files:
        image_path = os.path.join(args.instance_data_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        crop_size = min(width, height)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        left, top = interactive_crop_position(image_cv, crop_size, window_name)
        right = left + crop_size
        bottom = top + crop_size

        cropped_image = image.crop((left, top, right, bottom))
        cropped_image = cropped_image.resize(
            (args.resolution, args.resolution),
            Image.Resampling.LANCZOS,
        )

        output_path = os.path.join(output_dir, image_file)
        cropped_image.save(output_path)

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
import argparse
import os
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image


@dataclass
class CropSelection:
    left: int
    top: int
    rotation: int


class Navigation(Enum):
    PREVIOUS = "previous"
    NEXT = "next"


def navigation_from_key(key):
    if key in {81, 65361, 1113937, 2424832, 63234}:
        return Navigation.PREVIOUS
    if key in {83, 65363, 1113939, 2555904, 63235}:
        return Navigation.NEXT
    return None


def find_available_image(image_index, step, image_count, completed_indices):
    candidate = image_index + step
    while 0 <= candidate < image_count:
        if candidate not in completed_indices:
            return candidate
        candidate += step
    return image_index


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Interactive crop.")
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
    current_image = image_cv.copy()
    rotation = 0
    max_display = 1200

    while True:
        height, width = current_image.shape[:2]

        scale = min(1.0, max_display / max(width, height))
        disp_w, disp_h = max(1, int(width * scale)), max(1, int(height * scale))
        disp_crop_size = int(crop_size * scale)

        display_base = cv2.resize(current_image, (disp_w, disp_h))

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
            key = cv2.waitKeyEx(20)
            if key == ord('r') or key == ord('R'):
                rotation = (rotation + 90) % 360
                current_image = rotate_image_cv(current_image, 90)
                crop_size = min(current_image.shape[:2])
                break
            navigation = navigation_from_key(key)
            if navigation is not None:
                return navigation
            if key == ord("\r"):  # enter
                left = int(pos[0] / scale)
                top = int(pos[1] / scale)
                left = max(0, min(left, width - crop_size))
                top = max(0, min(top, height - crop_size))
                return CropSelection(left, top, rotation)
            elif key == ord("\x1b"):  # esc
                pos[0] = (disp_w - disp_crop_size) // 2
                pos[1] = (disp_h - disp_crop_size) // 2
                continue
            elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit(0)


def rotate_image_cv(image_cv, angle_degrees):
    angle = angle_degrees % 360
    if angle == 0:
        return image_cv.copy()
    if angle == 90:
        return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image_cv, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle_degrees} degrees")


def main():
    args = parse_args()

    if args.instance_data_dir is not None:
        output_dir = os.path.join(args.instance_data_dir, "cropped")
        image_files = sorted(
            f for f in os.listdir(args.instance_data_dir)
            if f.lower().endswith(".jpg")
        )
    elif args.instance_data is not None:
        output_dir = os.path.join(os.path.dirname(args.instance_data), "cropped")
        image_files = [args.instance_data]
    else:
        raise ValueError("Either --instance_data_dir or --instance_data must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    window_name = "Drag crop box, press ENTER to confirm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 1200)

    image_index = 0
    completed_indices = set()
    while image_index < len(image_files):
        image_file = image_files[image_index]
        if args.instance_data_dir is not None:
            image_path = os.path.join(args.instance_data_dir, image_file)
        else:
            image_path = args.instance_data
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        crop_size = min(width, height)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        crop_result = interactive_crop_position(image_cv, crop_size, window_name)
        if crop_result is Navigation.PREVIOUS:
            image_index = find_available_image(
                image_index, -1, len(image_files), completed_indices
            )
            continue
        if crop_result is Navigation.NEXT:
            image_index = find_available_image(
                image_index, 1, len(image_files), completed_indices
            )
            continue

        left, top, rotation = crop_result.left, crop_result.top, crop_result.rotation
        rotated_image_cv = rotate_image_cv(image_cv, rotation)
        rotated_crop_size = min(rotated_image_cv.shape[:2])
        right = left + rotated_crop_size
        bottom = top + rotated_crop_size

        rotated_image = Image.fromarray(cv2.cvtColor(rotated_image_cv, cv2.COLOR_BGR2RGB))
        cropped_image = rotated_image.crop((left, top, right, bottom))
        cropped_image = cropped_image.resize(
            (args.resolution, args.resolution),
            Image.Resampling.LANCZOS,
        )

        output_path = os.path.join(output_dir, os.path.basename(image_file))
        cropped_image.save(output_path)
        completed_indices.add(image_index)
        next_index = find_available_image(
            image_index, 1, len(image_files), completed_indices
        )
        if next_index == image_index:
            next_index = find_available_image(
                image_index, -1, len(image_files), completed_indices
            )
        if next_index == image_index:
            break
        image_index = next_index

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
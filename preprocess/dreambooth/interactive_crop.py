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


class KeyAction(Enum):
    NONE = "none"
    ROTATE = "rotate"
    PREVIOUS = "previous"
    NEXT = "next"
    CONFIRM = "confirm"
    RESET = "reset"


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
            action = action_from_keypress(key)
            match action:
                case KeyAction.ROTATE:
                    rotation = (rotation + 90) % 360
                    current_image = cv2.rotate(current_image, cv2.ROTATE_90_CLOCKWISE)
                    crop_size = min(current_image.shape[:2])
                    break
                case KeyAction.PREVIOUS:
                    return Navigation.PREVIOUS
                case KeyAction.NEXT:
                    return Navigation.NEXT
                case KeyAction.CONFIRM:
                    left = int(pos[0] / scale)
                    top = int(pos[1] / scale)
                    left = max(0, min(left, width - crop_size))
                    top = max(0, min(top, height - crop_size))
                    return CropSelection(left, top, rotation)
                case KeyAction.RESET:
                    pos[0] = (disp_w - disp_crop_size) // 2
                    pos[1] = (disp_h - disp_crop_size) // 2
                case KeyAction.NONE:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        raise SystemExit(0)


def action_from_keypress(key):
    if key in (ord("r"), ord("R")):
        return KeyAction.ROTATE
    if key in {81, 65361, 1113937, 2424832, 63234}:
        return KeyAction.PREVIOUS
    if key in {83, 65363, 1113939, 2555904, 63235}:
        return KeyAction.NEXT
    if key == ord("\r"):
        return KeyAction.CONFIRM
    if key == ord("\x1b"):
        return KeyAction.RESET
    return KeyAction.NONE


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

    available_indices = list(range(len(image_files)))
    image_position = 0
    while available_indices:
        image_index = available_indices[image_position]
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
            image_position = max(0, image_position - 1)
            continue
        if crop_result is Navigation.NEXT:
            image_position = min(len(available_indices) - 1, image_position + 1)
            continue

        left, top, rotation = crop_result.left, crop_result.top, crop_result.rotation
        match rotation:
            case 0:
                rotated_image_cv = image_cv
            case 90:
                rotated_image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
            case 180:
                rotated_image_cv = cv2.rotate(image_cv, cv2.ROTATE_180)
            case 270:
                rotated_image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        available_indices.pop(image_position)
        if image_position == len(available_indices):
            image_position -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
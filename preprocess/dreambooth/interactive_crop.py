import argparse
import os
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image


class Rotation(Enum):
    CLOCKWISE_0 = 0
    CLOCKWISE_90 = 90
    CLOCKWISE_180 = 180
    CLOCKWISE_270 = 270


class KeyAction(Enum):
    NONE = "none"
    ROTATE = "rotate"
    PREVIOUS = "previous"
    NEXT = "next"
    CONFIRM = "confirm"
    RESET = "reset"
    SKIP = "skip"
    BACK = "back"


@dataclass
class CropSelection:
    left: int
    top: int
    rotation: Rotation


@dataclass
class PrecropSelection:
    bounds: tuple[int, int, int, int]
    rotation: Rotation


LEFT_ARROW_KEYS = frozenset({81, 65361, 1113937, 2424832, 63234})
RIGHT_ARROW_KEYS = frozenset({83, 65363, 1113939, 2555904, 63235})
ROTATE_KEYS = frozenset({ord("r"), ord("R")})
CONFIRM_KEYS = frozenset({ord("\r"), 10})
RESET_KEYS = frozenset({ord("\x1b")})
SKIP_KEYS = frozenset({ord("s"), ord("S")})
BACK_KEYS = frozenset({ord("b"), ord("B")})


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


def rotate_image(image_cv, rotation):
    match rotation:
        case Rotation.CLOCKWISE_0:
            return image_cv
        case Rotation.CLOCKWISE_90:
            return cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
        case Rotation.CLOCKWISE_180:
            return cv2.rotate(image_cv, cv2.ROTATE_180)
        case Rotation.CLOCKWISE_270:
            return cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_bounds_clockwise(bounds, image_height):
    left, top, right, bottom = bounds
    return image_height - bottom, left, image_height - top, right


def draw_instructions(frame, lines):
    line_height = 25
    banner_height = line_height * len(lines) + 12
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(banner, 0.72, frame, 0.28, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def interactive_precrop(image_cv, window_name, initial_bounds=None):
    current_image = image_cv.copy()
    rotation = Rotation.CLOCKWISE_0
    max_display = 1200
    has_custom_selection = initial_bounds is not None and initial_bounds != (
        0,
        0,
        image_cv.shape[1],
        image_cv.shape[0],
    )
    selection = list(
        initial_bounds
        if initial_bounds is not None
        else (0, 0, image_cv.shape[1], image_cv.shape[0])
    )

    while True:
        height, width = current_image.shape[:2]
        scale = min(1.0, max_display / max(width, height))
        disp_w, disp_h = max(1, int(width * scale)), max(1, int(height * scale))
        display_base = cv2.resize(current_image, (disp_w, disp_h))
        dragging = {"mode": None, "start": (0, 0), "orig": None, "corner": None}

        def clamp_selection():
            selection[0] = max(0, min(selection[0], width - 1))
            selection[1] = max(0, min(selection[1], height - 1))
            selection[2] = max(selection[0] + 1, min(selection[2], width))
            selection[3] = max(selection[1] + 1, min(selection[3], height))

        def on_mouse(event, x, y, flags, param):
            nonlocal has_custom_selection
            image_x = max(0, min(int(x / scale), width - 1))
            image_y = max(0, min(int(y / scale), height - 1))
            if event == cv2.EVENT_LBUTTONDOWN:
                left, top, right, bottom = selection
                handle_radius = max(12, int(12 / scale))
                near_top_left = (
                    abs(image_x - left) <= handle_radius
                    and abs(image_y - top) <= handle_radius
                )
                near_bottom_right = (
                    abs(image_x - right) <= handle_radius
                    and abs(image_y - bottom) <= handle_radius
                )
                inside = left <= image_x <= right and top <= image_y <= bottom
                if near_top_left:
                    mode, corner = "resize", "top_left"
                elif near_bottom_right:
                    mode, corner = "resize", "bottom_right"
                elif inside and has_custom_selection:
                    mode, corner = "move", None
                else:
                    mode, corner = "draw", None
                    selection[:] = [image_x, image_y, image_x + 1, image_y + 1]
                    has_custom_selection = True
                dragging.update(
                    mode=mode,
                    start=(image_x, image_y),
                    orig=tuple(selection),
                    corner=corner,
                )
            elif event == cv2.EVENT_MOUSEMOVE and dragging["mode"] is not None:
                start_x, start_y = dragging["start"]
                dx, dy = image_x - start_x, image_y - start_y
                if dragging["mode"] == "draw":
                    selection[:] = [
                        min(start_x, image_x),
                        min(start_y, image_y),
                        max(start_x, image_x) + 1,
                        max(start_y, image_y) + 1,
                    ]
                elif dragging["mode"] == "move":
                    left, top, right, bottom = dragging["orig"]
                    box_width, box_height = right - left, bottom - top
                    left = max(0, min(left + dx, width - box_width))
                    top = max(0, min(top + dy, height - box_height))
                    selection[:] = [left, top, left + box_width, top + box_height]
                elif dragging["corner"] == "bottom_right":
                    left, top, _, _ = dragging["orig"]
                    selection[:] = [left, top, image_x + 1, image_y + 1]
                else:
                    _, _, right, bottom = dragging["orig"]
                    selection[:] = [image_x, image_y, right, bottom]
                clamp_selection()
            elif event == cv2.EVENT_LBUTTONUP:
                dragging["mode"] = None

        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            frame = (display_base * 0.4).astype(np.uint8)
            left, top, right, bottom = [int(value * scale) for value in selection]
            frame[top:bottom, left:right] = display_base[top:bottom, left:right]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
            cv2.rectangle(frame, (left - 5, top - 5), (left + 5, top + 5), (0, 0, 0), -1)
            cv2.rectangle(
                frame,
                (right - 5, bottom - 5),
                (right + 5, bottom + 5),
                (0, 0, 0),
                -1,
            )
            frame = draw_instructions(
                frame,
                [
                    "1/2 - Trim image",
                    "Drag to draw or move; drag a corner to resize",
                    "ENTER continue | S skip | R rotate | arrows navigate | ESC reset",
                ],
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKeyEx(20)
            action = action_from_keypress(key)
            match action:
                case KeyAction.ROTATE:
                    rotation = Rotation((rotation.value + 90) % 360)
                    selection[:] = rotate_bounds_clockwise(selection, height)
                    current_image = cv2.rotate(current_image, cv2.ROTATE_90_CLOCKWISE)
                    break
                case KeyAction.PREVIOUS:
                    return KeyAction.PREVIOUS
                case KeyAction.NEXT:
                    return KeyAction.NEXT
                case KeyAction.CONFIRM:
                    return PrecropSelection(tuple(selection), rotation)
                case KeyAction.SKIP:
                    return PrecropSelection((0, 0, width, height), rotation)
                case KeyAction.RESET:
                    selection[:] = [0, 0, width, height]
                    has_custom_selection = False
                case KeyAction.NONE:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        raise SystemExit(0)


def interactive_crop_position(
    image_cv,
    bounds,
    window_name,
    initial_position=None,
    initial_rotation=Rotation.CLOCKWISE_0,
):
    current_image = image_cv.copy()
    rotation = Rotation.CLOCKWISE_0
    max_display = 1200
    bounds = tuple(bounds)
    for _ in range(initial_rotation.value // 90):
        height, width = current_image.shape[:2]
        bounds = rotate_bounds_clockwise(bounds, height)
        current_image = cv2.rotate(current_image, cv2.ROTATE_90_CLOCKWISE)
        rotation = Rotation((rotation.value + 90) % 360)
    left, top, right, bottom = bounds
    crop_size = min(right - left, bottom - top)
    if initial_position is None:
        crop_left = left + (right - left - crop_size) // 2
        crop_top = top + (bottom - top - crop_size) // 2
    else:
        crop_left, crop_top = initial_position

    while True:
        height, width = current_image.shape[:2]
        scale = min(1.0, max_display / max(width, height))
        disp_w, disp_h = max(1, int(width * scale)), max(1, int(height * scale))
        disp_crop_size = max(1, int(crop_size * scale))
        display_base = cv2.resize(current_image, (disp_w, disp_h))
        dragging = {"active": False, "start": (0, 0), "orig": (0, 0)}

        def clamp():
            nonlocal crop_left, crop_top
            crop_left = max(bounds[0], min(crop_left, bounds[2] - crop_size))
            crop_top = max(bounds[1], min(crop_top, bounds[3] - crop_size))

        def on_mouse(event, x, y, flags, param):
            nonlocal crop_left, crop_top
            image_x = int(x / scale)
            image_y = int(y / scale)
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging["active"] = True
                dragging["start"] = (image_x, image_y)
                dragging["orig"] = (crop_left, crop_top)
            elif event == cv2.EVENT_MOUSEMOVE and dragging["active"]:
                dx = image_x - dragging["start"][0]
                dy = image_y - dragging["start"][1]
                crop_left, crop_top = dragging["orig"][0] + dx, dragging["orig"][1] + dy
                clamp()
            elif event == cv2.EVENT_LBUTTONUP:
                dragging["active"] = False

        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            frame = (display_base * 0.4).astype(np.uint8)
            bound_left, bound_top, bound_right, bound_bottom = [
                int(value * scale) for value in bounds
            ]
            frame[bound_top:bound_bottom, bound_left:bound_right] = \
                display_base[bound_top:bound_bottom, bound_left:bound_right]
            display_left, display_top = int(crop_left * scale), int(crop_top * scale)
            frame[
                display_top:display_top + disp_crop_size,
                display_left:display_left + disp_crop_size,
            ] = display_base[
                display_top:display_top + disp_crop_size,
                display_left:display_left + disp_crop_size,
            ]
            cv2.rectangle(
                frame,
                (bound_left, bound_top),
                (bound_right, bound_bottom),
                (255, 190, 0),
                2,
            )
            cv2.rectangle(
                frame,
                (display_left, display_top),
                (display_left + disp_crop_size, display_top + disp_crop_size),
                (0, 0, 0),
                2,
            )
            frame = draw_instructions(
                frame,
                [
                    "2/2 - Position fixed square crop",
                    "Drag to position the square",
                    "ENTER preview | B back | R rotate | arrows navigate | ESC reset",
                ],
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKeyEx(20)
            action = action_from_keypress(key)
            match action:
                case KeyAction.ROTATE:
                    bounds = rotate_bounds_clockwise(bounds, height)
                    crop_left, crop_top, _, _ = rotate_bounds_clockwise(
                        (crop_left, crop_top, crop_left + crop_size, crop_top + crop_size),
                        height,
                    )
                    rotation = Rotation((rotation.value + 90) % 360)
                    current_image = cv2.rotate(current_image, cv2.ROTATE_90_CLOCKWISE)
                    break
                case KeyAction.PREVIOUS:
                    return KeyAction.PREVIOUS
                case KeyAction.NEXT:
                    return KeyAction.NEXT
                case KeyAction.CONFIRM:
                    return CropSelection(crop_left, crop_top, rotation)
                case KeyAction.BACK:
                    return KeyAction.BACK
                case KeyAction.RESET:
                    crop_left = bounds[0] + (bounds[2] - bounds[0] - crop_size) // 2
                    crop_top = bounds[1] + (bounds[3] - bounds[1] - crop_size) // 2
                case KeyAction.NONE:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        raise SystemExit(0)


def interactive_crop_preview(image, window_name):
    preview = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    max_display = 1200
    height, width = preview.shape[:2]
    scale = min(1.0, max_display / max(width, height))
    if scale < 1.0:
        preview = cv2.resize(
            preview,
            (max(1, int(width * scale)), max(1, int(height * scale))),
        )
    preview = draw_instructions(
        preview,
        ["Final preview", "ENTER save | B adjust crop | arrows navigate"],
    )
    while True:
        cv2.imshow(window_name, preview)
        action = action_from_keypress(cv2.waitKeyEx(20))
        if action in (KeyAction.CONFIRM, KeyAction.BACK, KeyAction.PREVIOUS, KeyAction.NEXT):
            return action
        if action is KeyAction.NONE and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            raise SystemExit(0)


def action_from_keypress(key):
    if key in ROTATE_KEYS:
        return KeyAction.ROTATE
    if key in LEFT_ARROW_KEYS:
        return KeyAction.PREVIOUS
    if key in RIGHT_ARROW_KEYS:
        return KeyAction.NEXT
    if key in CONFIRM_KEYS:
        return KeyAction.CONFIRM
    if key in RESET_KEYS:
        return KeyAction.RESET
    if key in SKIP_KEYS:
        return KeyAction.SKIP
    if key in BACK_KEYS:
        return KeyAction.BACK
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

    window_name = "Interactive image crop"
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
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        working_image_cv = image_cv
        precrop_bounds = None
        navigate_action = None
        saved = False

        while not saved and navigate_action is None:
            precrop_result = interactive_precrop(
                working_image_cv,
                window_name,
                initial_bounds=precrop_bounds,
            )
            if precrop_result in (KeyAction.PREVIOUS, KeyAction.NEXT):
                navigate_action = precrop_result
                break

            working_image_cv = rotate_image(working_image_cv, precrop_result.rotation)
            precrop_bounds = precrop_result.bounds
            initial_crop_position = None
            initial_crop_rotation = Rotation.CLOCKWISE_0

            while True:
                crop_result = interactive_crop_position(
                    working_image_cv,
                    precrop_bounds,
                    window_name,
                    initial_position=initial_crop_position,
                    initial_rotation=initial_crop_rotation,
                )
                if crop_result is KeyAction.BACK:
                    break
                if crop_result in (KeyAction.PREVIOUS, KeyAction.NEXT):
                    navigate_action = crop_result
                    break

                rotated_image_cv = rotate_image(working_image_cv, crop_result.rotation)
                crop_size = min(
                    precrop_bounds[2] - precrop_bounds[0],
                    precrop_bounds[3] - precrop_bounds[1],
                )
                cropped_image = Image.fromarray(
                    cv2.cvtColor(rotated_image_cv, cv2.COLOR_BGR2RGB)
                ).crop(
                    (
                        crop_result.left,
                        crop_result.top,
                        crop_result.left + crop_size,
                        crop_result.top + crop_size,
                    )
                )
                cropped_image = cropped_image.resize(
                    (args.resolution, args.resolution),
                    Image.Resampling.LANCZOS,
                )
                preview_action = interactive_crop_preview(cropped_image, window_name)
                if preview_action is KeyAction.BACK:
                    initial_crop_position = (crop_result.left, crop_result.top)
                    initial_crop_rotation = crop_result.rotation
                    continue
                if preview_action in (KeyAction.PREVIOUS, KeyAction.NEXT):
                    navigate_action = preview_action
                    break

                output_path = os.path.join(output_dir, os.path.basename(image_file))
                cropped_image.save(output_path)
                saved = True
                break

            if navigate_action is not None:
                break

        if navigate_action is KeyAction.PREVIOUS:
            image_position = (image_position - 1) % len(available_indices)
            continue
        if navigate_action is KeyAction.NEXT:
            image_position = (image_position + 1) % len(available_indices)
            continue

        available_indices.pop(image_position)
        if image_position == len(available_indices):
            image_position -= 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Batch-create a magnified inset for PNG images of the same size."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


WINDOW_ROI = "Select area, then press ENTER or SPACE"
WINDOW_PREVIEW = "Preview - ENTER: continue, ESC: cancel"
Box = Tuple[int, int, int, int]


def natural_key(path: Path) -> list[object]:
    """Sort image2.png before image10.png."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def find_png_files(folder: Path, recursive: bool, suffix: str) -> list[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    files = [
        path for path in iterator
        if path.is_file()
        and path.suffix.lower() == ".png"
        and not path.stem.endswith(suffix)
    ]
    return sorted(files, key=natural_key)


def read_image(path: Path) -> np.ndarray:
    """Read PNG files reliably even when a path contains non-ASCII characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"Unable to read file: {path}") from exc

    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Not a valid PNG image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"Unsupported number of image channels: {path}")
    return image


def write_png(path: Path, image: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {path}\n"
            "Add --overwrite to the command to replace existing files."
        )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise RuntimeError(f"Unable to save file: {path}") from exc


def fit_for_display(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    factor = min(1.0, max_width / width, max_height / height)
    if factor == 1.0:
        return image.copy(), factor
    resized = cv2.resize(
        image,
        (max(1, round(width * factor)), max(1, round(height * factor))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, factor


def select_roi(image: np.ndarray, max_width: int, max_height: int) -> Box:
    display, factor = fit_for_display(image, max_width, max_height)
    print("Drag to select the area to magnify, then press Enter or Space to confirm; press Esc to cancel.")
    try:
        selected = cv2.selectROI(
            WINDOW_ROI, display, showCrosshair=True, fromCenter=False
        )
        cv2.destroyWindow(WINDOW_ROI)
    except cv2.error as exc:
        raise RuntimeError(
            "Unable to open the GUI window. Run this program in a desktop environment "
            "or use --roi x,y,w,h to specify the region directly."
        ) from exc

    x, y, width, height = (int(value) for value in selected)
    if width <= 0 or height <= 0:
        raise RuntimeError("No valid region was selected; operation cancelled.")

    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width - 1, round(x / factor)))
    y1 = max(0, min(image_height - 1, round(y / factor)))
    x2 = max(x1 + 1, min(image_width, round((x + width) / factor)))
    y2 = max(y1 + 1, min(image_height, round((y + height) / factor)))
    return x1, y1, x2 - x1, y2 - y1


def parse_roi(text: str) -> Box:
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must use the format x,y,w,h") from exc
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must use the format x,y,w,h, and width and height must be positive")
    return values  # type: ignore[return-value]


def validate_roi(roi: Box, image_width: int, image_height: int) -> None:
    x, y, width, height = roi
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {roi} is outside the image bounds (image size: {image_width}x{image_height})"
        )


def inset_size(
    roi: Box,
    scale: float,
    image_width: int,
    image_height: int,
    margin: int,
) -> tuple[int, int]:
    _, _, roi_width, roi_height = roi
    target_width = round(roi_width * scale)
    target_height = round(roi_height * scale)
    max_width = image_width - 2 * margin
    max_height = image_height - 2 * margin
    if max_width <= 0 or max_height <= 0:
        raise ValueError("The margin is too large to fit the magnified inset.")

    fit = min(1.0, max_width / target_width, max_height / target_height)
    target_width = max(1, round(target_width * fit))
    target_height = max(1, round(target_height * fit))
    if target_width <= roi_width or target_height <= roi_height:
        raise ValueError(
            "The selected region is too large to enlarge within the image. "
            "Select a smaller region or decrease --margin."
        )
    return target_width, target_height


def overlap_area(first: Box, second: Box) -> int:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    return width * height


def choose_destination(
    roi: Box,
    inset_width: int,
    inset_height: int,
    image_width: int,
    image_height: int,
    margin: int,
    placement: str,
) -> Box:
    candidates = {
        "top-left": (margin, margin, inset_width, inset_height),
        "top-right": (
            image_width - margin - inset_width,
            margin,
            inset_width,
            inset_height,
        ),
        "bottom-left": (
            margin,
            image_height - margin - inset_height,
            inset_width,
            inset_height,
        ),
        "bottom-right": (
            image_width - margin - inset_width,
            image_height - margin - inset_height,
            inset_width,
            inset_height,
        ),
    }
    if placement != "auto":
        return candidates[placement]

    # Prefer the corner that covers the least of the selected source area.
    priority = ("top-right", "top-left", "bottom-right", "bottom-left")
    return min((candidates[name] for name in priority), key=lambda box: overlap_area(roi, box))


def parse_rgb(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color must use the format R,G,B, for example 255,0,0") from exc
    if len(values) != 3 or any(value < 0 or value > 255 for value in values):
        raise argparse.ArgumentTypeError("R, G, and B must be integers between 0 and 255")
    red, green, blue = values
    return blue, green, red


def drawing_color(image: np.ndarray, bgr: tuple[int, int, int]) -> tuple[float, ...]:
    if np.issubdtype(image.dtype, np.integer):
        factor = np.iinfo(image.dtype).max / 255.0
    else:
        factor = 1.0 / 255.0 if float(np.max(image)) <= 1.0 else 1.0
    values = tuple(round(value * factor) for value in bgr)
    if image.shape[2] == 4:
        alpha = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
        return values + (alpha,)
    return values


def draw_connectors(
    image: np.ndarray,
    source: Box,
    destination: Box,
    color: tuple[float, ...],
    thickness: int,
) -> None:
    sx, sy, sw, sh = source
    dx, dy, dw, dh = destination
    source_center = (sx + sw / 2, sy + sh / 2)
    destination_center = (dx + dw / 2, dy + dh / 2)
    delta_x = destination_center[0] - source_center[0]
    delta_y = destination_center[1] - source_center[1]

    if abs(delta_x) >= abs(delta_y):
        if delta_x >= 0:
            source_points = ((sx + sw - 1, sy), (sx + sw - 1, sy + sh - 1))
            destination_points = ((dx, dy), (dx, dy + dh - 1))
        else:
            source_points = ((sx, sy), (sx, sy + sh - 1))
            destination_points = ((dx + dw - 1, dy), (dx + dw - 1, dy + dh - 1))
    else:
        if delta_y >= 0:
            source_points = ((sx, sy + sh - 1), (sx + sw - 1, sy + sh - 1))
            destination_points = ((dx, dy), (dx + dw - 1, dy))
        else:
            source_points = ((sx, sy), (sx + sw - 1, sy))
            destination_points = ((dx, dy + dh - 1), (dx + dw - 1, dy + dh - 1))

    for start, end in zip(source_points, destination_points):
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def make_composite(
    image: np.ndarray,
    roi: Box,
    destination: Box,
    bgr: tuple[int, int, int],
    border_width: int,
    line_width: int,
    connectors: bool,
) -> np.ndarray:
    x, y, width, height = roi
    dx, dy, destination_width, destination_height = destination
    crop = image[y:y + height, x:x + width]
    interpolation = cv2.INTER_CUBIC
    enlarged = cv2.resize(
        crop,
        (destination_width, destination_height),
        interpolation=interpolation,
    )

    result = image.copy()
    color = drawing_color(result, bgr)
    cv2.rectangle(
        result,
        (x, y),
        (x + width - 1, y + height - 1),
        color,
        border_width,
        cv2.LINE_AA,
    )
    if connectors:
        draw_connectors(result, roi, destination, color, line_width)

    result[dy:dy + destination_height, dx:dx + destination_width] = enlarged
    cv2.rectangle(
        result,
        (dx, dy),
        (dx + destination_width - 1, dy + destination_height - 1),
        color,
        border_width,
        cv2.LINE_AA,
    )
    return result


def confirm_preview(image: np.ndarray, max_width: int, max_height: int) -> None:
    display, _ = fit_for_display(image, max_width, max_height)
    print("Preview: press Enter or Space to start; press Esc or Q to cancel.")
    try:
        cv2.imshow(WINDOW_PREVIEW, display)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (10, 13, 32):
                break
            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError("Operation cancelled.")
        cv2.destroyWindow(WINDOW_PREVIEW)
    except cv2.error as exc:
        raise RuntimeError(
            "Unable to open the preview window; add --no-preview to skip it."
        ) from exc


def edit_magnifier(
    image: np.ndarray,
    scale: float,
    placement: str,
    margin: int,
    bgr: tuple[int, int, int],
    border_width: int,
    line_width: int,
    connectors: bool,
    max_width: int,
    max_height: int,
    initial_roi: Optional[Box] = None,
) -> tuple[Box, Box, float]:
    """Select an ROI while previewing, moving, and resizing its inset."""
    image_height, image_width = image.shape[:2]
    display_factor = min(
        1.0,
        max_width / image_width,
        max_height / image_height,
    )
    roi = initial_roi
    destination: Optional[Box] = None
    mode: Optional[str] = None
    selection_start = (0, 0)
    selection_start_display = (0, 0)
    previous_roi: Optional[Box] = None
    previous_destination: Optional[Box] = None
    drag_offset = (0, 0)
    invalid_selection = False

    def display_to_image(x: int, y: int) -> tuple[int, int]:
        return (
            max(0, min(image_width - 1, round(x / display_factor))),
            max(0, min(image_height - 1, round(y / display_factor))),
        )

    def point_in_box(point: tuple[int, int], box: Box) -> bool:
        x, y = point
        bx, by, width, height = box
        return bx <= x < bx + width and by <= y < by + height

    def update_destination(reset_position: bool) -> None:
        nonlocal destination, invalid_selection
        if roi is None:
            destination = None
            return
        try:
            width, height = inset_size(
                roi, scale, image_width, image_height, margin
            )
        except ValueError:
            destination = None
            invalid_selection = True
            return

        invalid_selection = False
        if reset_position or destination is None:
            destination = choose_destination(
                roi,
                width,
                height,
                image_width,
                image_height,
                margin,
                placement,
            )
            return

        old_x, old_y, old_width, old_height = destination
        center_x = old_x + old_width / 2
        center_y = old_y + old_height / 2
        x = round(center_x - width / 2)
        y = round(center_y - height / 2)
        x = max(0, min(image_width - width, x))
        y = max(0, min(image_height - height, y))
        destination = (x, y, width, height)

    def update_selection(point: tuple[int, int]) -> None:
        nonlocal roi
        start_x, start_y = selection_start
        current_x, current_y = point
        left = min(start_x, current_x)
        top = min(start_y, current_y)
        right = max(start_x, current_x)
        bottom = max(start_y, current_y)
        roi = (left, top, right - left + 1, bottom - top + 1)
        update_destination(reset_position=True)

    def mouse_callback(
        event: int,
        x: int,
        y: int,
        _flags: int,
        _param: object,
    ) -> None:
        nonlocal mode
        nonlocal selection_start, selection_start_display
        nonlocal previous_roi, previous_destination, drag_offset
        nonlocal roi, destination

        point = display_to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if destination is not None and point_in_box(point, destination):
                dx, dy, _, _ = destination
                mode = "move"
                drag_offset = (point[0] - dx, point[1] - dy)
            else:
                mode = "select"
                previous_roi = roi
                previous_destination = destination
                selection_start = point
                selection_start_display = (x, y)
                update_selection(point)

        elif event == cv2.EVENT_MOUSEMOVE:
            if mode == "select":
                update_selection(point)
            elif mode == "move" and destination is not None:
                _, _, width, height = destination
                new_x = point[0] - drag_offset[0]
                new_y = point[1] - drag_offset[1]
                new_x = max(0, min(image_width - width, new_x))
                new_y = max(0, min(image_height - height, new_y))
                destination = (new_x, new_y, width, height)

        elif event == cv2.EVENT_LBUTTONUP:
            if mode == "select":
                moved = max(
                    abs(x - selection_start_display[0]),
                    abs(y - selection_start_display[1]),
                )
                if moved < 3:
                    roi = previous_roi
                    destination = previous_destination
                else:
                    update_selection(point)
            mode = None

    def render() -> np.ndarray:
        if roi is not None and destination is not None:
            frame = make_composite(
                image,
                roi,
                destination,
                bgr,
                border_width,
                line_width,
                connectors,
            )
        else:
            frame = image.copy()
            if roi is not None:
                x, y, width, height = roi
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width - 1, y + height - 1),
                    drawing_color(frame, bgr),
                    border_width,
                    cv2.LINE_AA,
                )

        display, _ = fit_for_display(frame, max_width, max_height)
        help_lines = [
            "Drag outside inset: select area | Drag inset: move",
            f"+/-: zoom ({scale:.2f}x) | R: reset position | Enter: apply | Esc: cancel",
        ]
        if invalid_selection:
            help_lines.append("Selection is too large to enlarge inside this image.")

        panel_height = 10 + 24 * len(help_lines)
        overlay = display.copy()
        cv2.rectangle(
            overlay, (0, 0), (display.shape[1], panel_height), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        for index, text in enumerate(help_lines):
            color = (80, 180, 255) if index == 2 else (255, 255, 255)
            cv2.putText(
                display,
                text,
                (10, 22 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
        return display

    if roi is not None:
        update_destination(reset_position=True)

    print(
        "Drag to select an area; drag the magnified inset to move it; "
        "press + or - to change the zoom; press Enter to confirm or Esc to cancel."
    )
    window_name = "Magnifier editor"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            cv2.imshow(window_name, render())
            key = cv2.waitKey(20)
            if key < 0:
                continue
            key &= 0xFF
            if key in (ord("+"), ord("=")):
                old_scale = scale
                old_destination = destination
                scale = round(scale + 0.25, 2)
                update_destination(reset_position=False)
                if destination is None and old_destination is not None:
                    scale = old_scale
                    destination = old_destination
            elif key in (ord("-"), ord("_")):
                old_scale = scale
                old_destination = destination
                scale = max(1.25, round(scale - 0.25, 2))
                update_destination(reset_position=False)
                if destination is None and old_destination is not None:
                    scale = old_scale
                    destination = old_destination
            elif key in (ord("r"), ord("R")):
                update_destination(reset_position=True)
            elif key in (10, 13, 32):
                if roi is not None and destination is not None:
                    return roi, destination, scale
            elif key in (27, ord("q"), ord("Q")):
                raise RuntimeError("Operation cancelled.")
    except cv2.error as exc:
        raise RuntimeError(
            "Unable to open the GUI window. Run this program in a desktop environment "
            "or use both --roi x,y,w,h and --no-preview."
        ) from exc
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add a magnified inset to a folder of same-sized PNG images."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing the source PNG images")
    parser.add_argument("output_dir", type=Path, help="Folder in which to save the results")
    parser.add_argument("--scale", type=float, default=2.0, help="Zoom factor; default: 2.0")
    parser.add_argument(
        "--placement",
        choices=("auto", "top-left", "top-right", "bottom-left", "bottom-right"),
        default="auto",
        help="Inset position; default: choose the corner with the least overlap",
    )
    parser.add_argument("--margin", type=int, default=20, help="Distance between the inset and image edge")
    parser.add_argument("--border-width", type=int, default=3, help="Border width")
    parser.add_argument("--line-width", type=int, default=2, help="Connector line width")
    parser.add_argument(
        "--color",
        type=parse_rgb,
        default=parse_rgb("255,0,0"),
        metavar="R,G,B",
        help="Border color; default: red (255,0,0)",
    )
    parser.add_argument("--suffix", default="_magnified", help="Output filename suffix")
    parser.add_argument("--roi", type=parse_roi, help="Specify x,y,w,h directly")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    parser.add_argument("--no-connectors", action="store_true", help="Do not draw connector lines")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not open the interactive editor when used with --roi",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--display-width", type=int, default=1400, help=argparse.SUPPRESS)
    parser.add_argument("--display-height", type=int, default=900, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scale <= 1.0:
        raise ValueError("--scale must be greater than 1.0")
    if args.margin < 0 or args.border_width <= 0 or args.line_width <= 0:
        raise ValueError("The margin cannot be negative, and line widths must be positive")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_dir}")

    files = find_png_files(args.input_dir, args.recursive, args.suffix)
    if not files:
        raise FileNotFoundError(f"No PNG images found in input folder: {args.input_dir}")

    first_image = read_image(files[0])
    image_height, image_width = first_image.shape[:2]
    print(f"First image: {files[0].name} ({image_width}x{image_height})")

    roi = args.roi
    if roi is not None:
        validate_roi(roi, image_width, image_height)

    if roi is None or not args.no_preview:
        roi, destination, args.scale = edit_magnifier(
            image=first_image,
            scale=args.scale,
            placement=args.placement,
            margin=args.margin,
            bgr=args.color,
            border_width=args.border_width,
            line_width=args.line_width,
            connectors=not args.no_connectors,
            max_width=args.display_width,
            max_height=args.display_height,
            initial_roi=roi,
        )
    else:
        destination_width, destination_height = inset_size(
            roi, args.scale, image_width, image_height, args.margin
        )
        destination = choose_destination(
            roi,
            destination_width,
            destination_height,
            image_width,
            image_height,
            args.margin,
            args.placement,
        )

    first_composite = make_composite(
        first_image,
        roi,
        destination,
        args.color,
        args.border_width,
        args.line_width,
        not args.no_connectors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(files, start=1):
        image = first_image if index == 1 else read_image(source_path)
        if image.shape[:2] != (image_height, image_width):
            raise ValueError(
                f"Image size mismatch: {source_path.name} is "
                f"{image.shape[1]}x{image.shape[0]}; expected {image_width}x{image_height}"
            )
        composite = first_composite if index == 1 else make_composite(
            image,
            roi,
            destination,
            args.color,
            args.border_width,
            args.line_width,
            not args.no_connectors,
        )
        output_path = args.output_dir / f"{source_path.stem}{args.suffix}.png"
        write_png(output_path, composite, args.overwrite)
        print(f"[{index}/{len(files)}] Saved: {output_path}")

    print(f"Done. Generated {len(files)} images.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


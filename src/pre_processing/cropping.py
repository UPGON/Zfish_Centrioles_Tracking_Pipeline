import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

import tifffile
from tqdm import tqdm


SUPPORTED_FORMATS = {"TZCYX", "TZYX", "ZCYX", "ZYX", "CYX", "YX"}


def _make_slice(start, end, length):
    return slice(0 if start is None else start, length if end is None else end)


def _normalize_format(img_format: str) -> str:
    fmt = img_format.upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {img_format}. Supported formats: {sorted(SUPPORTED_FORMATS)}")
    return fmt


def verify_input(image, x_start, y_start, x_end, y_end):
    if image.ndim < 2:
        raise ValueError(f"Expected image with at least 2 dimensions, got {image.shape}")

    img_height, img_width = image.shape[-2:]

    if not (0 <= x_start < x_end <= img_width):
        raise ValueError(
            f"x_start ({x_start}) must be >= 0, < x_end ({x_end}) and <= width ({img_width})"
        )
    if not (0 <= y_start < y_end <= img_height):
        raise ValueError(
            f"y_start ({y_start}) must be >= 0, < y_end ({y_end}) and <= height ({img_height})"
        )


def crop_image(
    img,
    img_format,
    t_start=None,
    t_end=None,
    z_start=None,
    z_end=None,
    y_start=None,
    y_end=None,
    x_start=None,
    x_end=None,
):
    img_format = _normalize_format(img_format)

    y_slice = _make_slice(y_start, y_end, img.shape[-2])
    x_slice = _make_slice(x_start, x_end, img.shape[-1])

    if img_format == "TZCYX":
        t_slice = _make_slice(t_start, t_end, img.shape[0])
        z_slice = _make_slice(z_start, z_end, img.shape[1])
        return img[t_slice, z_slice, :, y_slice, x_slice]

    if img_format == "TZYX":
        t_slice = _make_slice(t_start, t_end, img.shape[0])
        z_slice = _make_slice(z_start, z_end, img.shape[1])
        return img[t_slice, z_slice, y_slice, x_slice]

    if img_format == "ZCYX":
        z_slice = _make_slice(z_start, z_end, img.shape[0])
        return img[z_slice, :, y_slice, x_slice]

    if img_format == "ZYX":
        z_slice = _make_slice(z_start, z_end, img.shape[0])
        return img[z_slice, y_slice, x_slice]

    if img_format == "CYX":
        return img[:, y_slice, x_slice]

    if img_format == "YX":
        return img[y_slice, x_slice]

    raise ValueError(f"Unsupported image format: {img_format}")

def crop_file(
    input_path: Path,
    output_path: Path,
    img_format,
    t_start,
    t_end,
    z_start,
    z_end,
    y_start,
    y_end,
    x_start,
    x_end,
):
    with tifffile.TiffFile(input_path) as tif:
        img = tif.asarray()

        # ImageJ metadata
        imagej_metadata = tif.imagej_metadata or {}

        # Resolution tags (pixel size stored in TIFF tags)
        resolution    = None
        resolutionunit = None
        if tif.pages:
            page = tif.pages[0]
            x_res = page.tags.get("XResolution")
            y_res = page.tags.get("YResolution")
            res_unit = page.tags.get("ResolutionUnit")
            if x_res and y_res:
                resolution = (x_res.value, y_res.value)
            if res_unit:
                resolutionunit = res_unit.value

    cropped_img = crop_image(
        img,
        img_format,
        t_start,
        t_end,
        z_start,
        z_end,
        y_start,
        y_end,
        x_start,
        x_end,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / input_path.name
    if output_file.exists():
        print(f"Warning: Output file {output_file} already exists and will be overwritten.")

    tifffile.imwrite(
        output_file,
        cropped_img,
        imagej=True,
        metadata=imagej_metadata if imagej_metadata else None,
        resolution=resolution,
        resolutionunit=resolutionunit,
    )


def crop_file_task(args):
    return crop_file(*args)

def crop_data_set(
    input_path: Path,
    output_path: Path,
    img_format,
    t_start,
    t_end,
    z_start,
    z_end,
    y_start,
    y_end,
    x_start,
    x_end,
):
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]

    if not image_files:
        print("No images found in the provided directory.")
        sys.exit(0)

    tasks = [
        (
            img_path,
            output_path,
            img_format,
            t_start,
            t_end,
            z_start,
            z_end,
            y_start,
            y_end,
            x_start,
            x_end,
        )
        for img_path in image_files
    ]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(crop_file_task, task): task[0] for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Cropping images",
            unit="img",
        ):
            img_path = futures[future]
            try:
                future.result()
            except (ValueError, OSError) as exc:
                print(f"Error processing {img_path.name}: {exc}")


def cropping(
    input_path,
    output_path,
    img_format,
    t_start=None,
    t_end=None,
    z_start=None,
    z_end=None,
    y_start=None,
    y_end=None,
    x_start=None,
    x_end=None,
):
    start = time.time()

    if not input_path.exists():
        raise ValueError("The input path doesn't exists, please check the spelling")

    elif input_path.is_file():
        crop_file(
            input_path,
            output_path,
            img_format,
            t_start,
            t_end,
            z_start,
            z_end,
            y_start,
            y_end,
            x_start,
            x_end,
        )
    elif input_path.is_dir():
        crop_data_set(
            input_path,
            output_path,
            img_format,
            t_start,
            t_end,
            z_start,
            z_end,
            y_start,
            y_end,
            x_start,
            x_end,
        )
    else:
        raise FileNotFoundError(f"{input_path} is neither a file nor a directory.")

    print("Successfully cropped images")
    print(f"Operation time: {time.time() - start:.6f}s")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Crop the given image using provided coordinates.")
    parser.add_argument("--input_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument(
        "--format",
        type=str,
        default="TZCYX",
        help="Input image format. Supported values: TZCYX, TZYX, ZCYX, ZYX, CYX, YX",
    )
    parser.add_argument("--t_start", type=int)
    parser.add_argument("--t_end", type=int)
    parser.add_argument("--z_start", type=int)
    parser.add_argument("--z_end", type=int)
    parser.add_argument("--y_start", type=int)
    parser.add_argument("--y_end", type=int)
    parser.add_argument("--x_start", type=int)
    parser.add_argument("--x_end", type=int)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        cropping(
            args.input_path,
            args.output_path,
            args.format,
            args.t_start,
            args.t_end,
            args.z_start,
            args.z_end,
            args.y_start,
            args.y_end,
            args.x_start,
            args.x_end,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


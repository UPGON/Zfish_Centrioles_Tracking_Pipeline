import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm
import traceback
import os
import multiprocessing

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def stack_images(
    input_path: Path,
    channel_names
):
    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print("No images found in the provided directory.")
        sys.exit(0)

    stacked_channels = []

    for channel_name in channel_names:
        channel_files =  [img for img in image_files if channel_name in img.stem]
        if not channel_files:
            raise ValueError(f"No images found for channel '{channel_name}' in the provided directory.")
        
        channel_files.sort()
        timelapse = len(channel_files)
        sample_img = tifffile.imread(channel_files[0])
        z, y, x = sample_img.shape
        stacked_channel = np.empty((timelapse, z, y, x), dtype=sample_img.dtype)
        
        for ti in tqdm(range(timelapse), desc=f"Stacking channel {channel_name}", unit="frame"):
            stacked_channel[ti] = tifffile.imread(channel_files[ti])
        stacked_channels.append(stacked_channel)

    return np.stack(stacked_channels, axis=1)

def stack_channel_images(image_files, channel_name):
    channel_files =  [img for img in image_files if channel_name in img.stem]
    if not channel_files:
        raise ValueError(f"No images found for channel '{channel_name}' in the provided directory.")
    
    channel_files.sort()

    timelapse = len(channel_files)
    sample_img = tifffile.memmap(channel_files[0])
    z, y, x = sample_img.shape
    stacked_channel = np.empty((timelapse, z, y, x), dtype=sample_img.dtype)
    
    for ti in range(timelapse):
        stacked_channel[ti] = tifffile.imread(channel_files[ti])
    
    print(f"Finished stacking channel {channel_name}. Shape: {stacked_channel.shape}")
    return stacked_channel


def stack_tack(args):
    return stack_channel_images(*args)


def stack_channels_images(
    input_path: Path,
    channel_names,
    max_workers=None
):
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]

    if not image_files:
        print("No images found in the provided directory.")
        sys.exit(0)

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, 10)
    

    tasks = [
        (
            image_files,
            channel_name
        )
        for channel_name in channel_names
    ]

    stacked_channels = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(stack_tack, task): task[1] for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Stacking the channel images",
            unit="img",
        ):
            channel_name = futures[future]
            try:
                stacked_channel = future.result()
                if stacked_channel is not None:
                    stacked_channels[channel_name] = stacked_channel
            except (ValueError, OSError) as exc:
                print(f"Error processing {channel_name}: {exc}")

    if stacked_channels:
        stacked_vol = np.stack([stacked_channels[channel_name] for channel_name in channel_names], axis=2)
    else:
        stacked_vol = np.array([])

    print(f"Finished stacking channels. Final shape: {stacked_vol.shape}")

    return stacked_vol


def stacking(
    input_path,
    output_path,
    channel_names,
    max_workers = None
):
    start = time.time()

    if not input_path.is_dir():
        raise ValueError(f"Expected input_path to be a directory, got {input_path}")
    if not output_path.suffix.lower() in IMAGE_EXTENSIONS:
        raise ValueError(f"Expected output_path to be a file, got {output_path}")
    
    channel_names = channel_names.split(" ")
    if(len(channel_names) == 0):
        raise ValueError("At least one channel name must be provided")
    
    stacked_vol = stack_channels_images(input_path, channel_names,max_workers)

    os.makedirs(output_path.parent, exist_ok=True)
    tifffile.imwrite(
        output_path,
        stacked_vol,
        imagej=True,
        metadata={"axes": "TZCYX"},
        compression = "zlib",
    )

    print("Successfully stacked images")
    print(f"Operation time: {time.time() - start:.6f}s")



def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Stack all images of a given folder into a single multi-channel image")
    parser.add_argument("--input_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--channel_names", required=True, type=str)
    parser.add_argument("--max_workers", type=int,
                       help="Number of parallel workers (default: CPU count - 1)")
    return parser


if __name__ == "__main__":
    """Command-line interface for stacking single 4D images (TZYX) into a single 5D image.
        The image should have its channel number in its file name

    Usage:
        python interpolation.py --input_path <path_to_input_image> --output_path <path_to_output_image> --scale <scaling_factor> [--fx <fx>] [--fy <fy>]

    Args:
        --input_path (str): The path of the image to be interpolated. The image must be in 3D with format (T,Z,Y,X).
        --output_path (str): The path where the interpolated image(s) should be saved (must be a directory).
        --scale (float): The scaling factor for interpolation (e.g., 0.75 for 75% of original size).
        --fx (int, optional): The scaling factor along the x-axis. Default is 1.
        --fy (int, optional): The scaling factor along the y-axis. Default is 1.
    """
    try:
        args = _build_arg_parser().parse_args()

        stacking(
            args.input_path,
            args.output_path,
            args.channel_names,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
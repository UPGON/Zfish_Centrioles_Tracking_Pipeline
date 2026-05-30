import sys
from pathlib import Path
import os
import argparse
import pathlib
import time
import tifffile
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log
from skimage.filters import threshold_yen, threshold_minimum
from skimage.measure import label
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback



project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization
from utils import utils

def apply_thresholding_algorithm(img, algorithm):
    """Apply blob detection algorithm.

    Args:
        img: Input 3D image (Z, Y, X)
        algorithm: 'dog' or 'log'
        min_sigma: Minimum sigma for Gaussian kernel
        max_sigma: Maximum sigma for Gaussian kernel
        threshold: Detection threshold

    Returns:
        Array of detected blobs (z, y, x, r)
    """
    algorithms = {"yen": threshold_yen, "min": threshold_minimum}

    if algorithm not in algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not recognized. Use 'yen' or 'min'.")

    return algorithms[algorithm](
        img
        )

def thresholding_2D(img):
    thresh_yen = threshold_yen(img)
    yen_img = img > thresh_yen

    return yen_img

def remove_area(img,area_max):
    filtered_img = img.copy()
    filtered_img = remove_small_objects(filtered_img, min_size=area_max) != 0

    return filtered_img
    

def thresholding(frame, algorithm, area_max =None):
    frame_res = np.empty(frame.shape)
    for z in range(frame.shape[0]):
        threshold = apply_thresholding_algorithm(frame[z], algorithm)
        yen_img = frame[z] > threshold
        yen_img = yen_img.astype(int)
        if area_max is not None:
            yen_img = remove_area(yen_img, area_max)

        yen_img[yen_img == 1] = 255
        frame_res[z] = yen_img
    return frame_res
        

def thresholding_task(args):
    frame_idx, frame, algorithm , area_max = args
    thresh_frame = thresholding(frame, algorithm, area_max)
    return frame_idx, thresh_frame


def process_frames_parallel(
    vol_c,
    algorithm,
    area_max,
    max_workers=None
):
    """Process multiple frames in parallel.
    
    Args:
        vol_c: Volume with frames to process (T, Z, Y, X)
        algorithm: Detection algorithm
        area_max: Maximum area for filtering
        max_workers: Number of parallel workers
    
    Returns:
        Tuple of (all_blobs_coords, composite)
    """
    t, z, y, x = vol_c.shape
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, t)
    
    print(f"Processing {t} frames using {max_workers} workers...")
    
    # Prepare tasks
    tasks = [
        (ti, vol_c[ti], algorithm, area_max)
        for ti in range(t)
    ]
    
    # Pre-allocate composite
    yen_vol = np.empty((t, z, y, x), dtype=np.uint8)

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(thresholding_task, task): task[0] 
                  for task in tasks}
        
        with tqdm(total=t, desc="Thresholding ", unit="frame") as pbar:
            for future in as_completed(futures):
                try:
                    frame_idx, yen_frames = future.result()
                    
                    # Store composite
                    if yen_frames is not None:
                        yen_vol[frame_idx] = yen_frames
                                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError in frame {futures[future]}: {e}")
                    pbar.update(1)

    return yen_vol


def save_results(
    thresh_vol,
    output_path,
    timepoint,
):
    """Save detection results."""
    print("Saving results...")

    # Save composite image
    tifffile.imwrite(
        output_path,
        thresh_vol,
        imagej=True,
        metadata={"axes": "TZCYX" if timepoint is None else "ZCYX"},
        compression="zlib",
    )

def threshold(
    input_path,
    channel_id,
    algorithm,
    area_max=None,
    timepoint=None,
    z_min=None,
    z_max=None,
    output_path = None,
    max_workers=None,
):
    """Main blob detection pipeline with parallel processing.

    Args:
        input_path: Path to input TIFF
        output_path: Output directory
        channel_id: Channel to process
        resolution: Voxel size [dz, dy, dx] in microns
        algorithm: 'dog' or 'log'
        threshold: Detection threshold
        min_sigma: Minimum sigma
        max_sigma: Maximum sigma
        area_max: Maximum area in microns² for filtering
        timepoint: Optional specific timepoint
        z_min: Minimum Z slice
        z_max: Maximum Z slice
        max_workers: Number of parallel workers
    """
    start_time = time.time()
    
    print(f"Loading {input_path}...")
    vol = tifffile.memmap(input_path)
    py_channel_id = channel_id - 1  # Convert to 0-based index
    utils.verify_input(vol, py_channel_id, timepoint, z_min, z_max)
    
    z, c, y, x = vol.shape[-4:]
    pixel_size = utils.get_pixel_size(input_path)
    
    # Convert area_max from microns² to pixels²
    area_max_px = None
    if area_max is not None:
        area_max_px = area_max / (pixel_size[1] * pixel_size[2])

    composite = None

    # Set Z range
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else z
    z_range = slice(z_min, z_max)

    # Process 4D volume (no time dimension)
    if vol.ndim == 4:
        print("Processing single frame (4D volume)...")
        thresh_vol = thresholding(
            vol[z_range, py_channel_id], area_max
        )
        composite = np.stack([vol[z_range,py_channel_id], thresh_vol], axis = 1)
        timepoint = 0

    # Process 5D volume (with time dimension)
    elif vol.ndim == 5:
        t = vol.shape[0]
        vol_c = vol[:, z_range, py_channel_id]

        # Single timepoint
        if timepoint is not None:
            print(f"Processing single timepoint T={timepoint}...")
            thresh_vol = thresholding(
                vol_c[timepoint], algorithm, area_max_px
            )
            composite = np.stack([vol_c[timepoint], thresh_vol],axis = 1)

        # All timepoints (PARALLEL)
        else:
            thresh_vol = process_frames_parallel(
                vol_c,
                algorithm,
                area_max_px,
                max_workers=max_workers,
            )
            composite = np.stack([vol_c,thresh_vol], axis = 2)

    # Save results
    save_results(
        composite,
        output_path,
        timepoint,
    )

    elapsed = time.time() - start_time
    print(f"Detection complete")
    print(f"Total time: {elapsed:.2f}s")
    if timepoint is None and vol.ndim == 5:
        print(f"  Average: {elapsed/vol.shape[0]:.2f}s per frame")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Blob detection with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Single timepoint
                python %(prog)s --input_path data.tif --output_path results/ \\
                --channel_id 0 --resolution 0.5 0.15 0.15 --algorithm dog \\
                --threshold 0.01 --timepoint 5

                # All timepoints (parallel)
                python %(prog)s --input_path data.tif --output_path results/ \\
                --channel_id 0 --resolution 0.5 0.15 0.15 --algorithm dog \\
                --threshold 0.01 --max_workers 8
                """
    )
    
    parser.add_argument("--input_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--channel_id", required=True, type=int,
                       help="Channel to process")
    parser.add_argument("--algorithm", required=True, choices=['yen', 'min'],
                       help="Detection algorithm")
    parser.add_argument("--area_max", type=float,
                       help="Maximum area in µm² for filtering")
    parser.add_argument("--timepoint", type=int,
                       help="Process single timepoint")
    parser.add_argument("--z_min", type=int,
                       help="Minimum Z slice")
    parser.add_argument("--z_max", type=int,
                       help="Maximum Z slice")
    parser.add_argument("--output_path", type=pathlib.Path,
                       help="Output directory")
    parser.add_argument("--max_workers", type=int,
                       help="Number of parallel workers (default: CPU count - 1)")
    
    
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        threshold(
            args.input_path,
            args.channel_id,
            args.algorithm,
            args.area_max,
            args.timepoint,
            args.z_min,
            args.z_max,
            args.output_path,
            args.max_workers,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
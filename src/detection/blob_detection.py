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
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization
from utils import utils


def apply_detection_algorithm(img, algorithm, min_sigma, max_sigma, threshold):
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
    algorithms = {"log": blob_log, "dog": blob_dog}

    if algorithm not in algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not recognized. Use 'dog' or 'log'.")

    return algorithms[algorithm](
        img, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
    )


def remove_large_area_element(frame, blob_center, area_max):
    """Remove blobs that fall inside large objects.
    # TODO: Do a zmax proj around the detected blob and remove it based on its area
    
    Args:
        frame: 3D image (Z, Y, X)
        blob_center: Array of blob coordinates (N, 4) [z, y, x, r]
        area_max: Maximum area threshold in pixels
    
    Returns:
        Filtered blob array
    """
    zframes_detected = np.unique(blob_center[:, 0]).astype(int)
    keep_mask = np.ones(len(blob_center), dtype=bool)

    for z in zframes_detected:
        thresh_yen = threshold_yen(frame[z])
        thresh_yen_img = frame[z] > thresh_yen
        img_labels = label(thresh_yen_img)
        large_obj = remove_small_objects(img_labels, min_size=area_max) != 0
        
        z_indices = np.where(blob_center[:, 0] == z)[0]

        for idx in z_indices:
            y, x = blob_center[idx][1:3].astype(int)
            if 0 <= y < frame.shape[1] and 0 <= x < frame.shape[2]:
                if large_obj[y, x]:
                    keep_mask[idx] = False

    return blob_center[keep_mask]


def frame_blob_detection(frame, algorithm, threshold, min_sigma, max_sigma, area_max):
    """Wrapper for parallel frame processing.
    
    Args:
        args: Tuple of (frame_idx, frame, algorithm, threshold, min_sigma, max_sigma, area_max)
    
    Returns:
        Tuple of (frame_idx, blobs_coords, composite)
    """
    # Detect blobs
    blobs_center = apply_detection_algorithm(
        frame, algorithm, min_sigma, max_sigma, threshold
    )
    
    # Convert sigma to radius
    blobs_center[:, -1] = blobs_center[:, -1] * np.sqrt(3)
    
    # Filter by area
    if area_max is not None:
        blobs_center = remove_large_area_element(frame, blobs_center, area_max)
    
    # Create visualization
    mask = visualization.create_circle_mask(blobs_center[:, :3], frame.shape)
    composite = np.stack([frame, mask], axis=1)
    
    return blobs_center, composite
        

def frame_blob_detection_task(args):
    frame_idx, frame, algorithm, threshold, min_sigma, max_sigma, area_max = args
    blobs_center, composite = frame_blob_detection(frame, algorithm, threshold, min_sigma, max_sigma, area_max)
    return frame_idx, blobs_center, composite


def process_frames_parallel(
    vol_c,
    algorithm,
    threshold,
    min_sigma,
    max_sigma,
    area_max,
    max_workers=None
):
    """Process multiple frames in parallel.
    
    Args:
        vol_c: Volume with frames to process (T, Z, Y, X)
        algorithm: Detection algorithm
        threshold: Detection threshold
        min_sigma: Minimum sigma
        max_sigma: Maximum sigma
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
        (ti, vol_c[ti], algorithm, threshold, min_sigma, max_sigma, area_max)
        for ti in range(t)
    ]
    
    # Pre-allocate composite
    composite = np.empty((t, z, 2, y, x), dtype=np.uint8)
    all_blobs_coords = []
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(frame_blob_detection_task, task): task[0] 
                  for task in tasks}
        
        with tqdm(total=t, desc="Detecting blobs", unit="frame") as pbar:
            for future in as_completed(futures):
                try:
                    frame_idx, blobs_coords, comp = future.result()
                    
                    # Store composite
                    if comp is not None:
                        composite[frame_idx] = comp
                    
                    # Add timepoint column to blobs
                    if len(blobs_coords) > 0:
                        blobs_with_t = np.column_stack([
                            np.full(len(blobs_coords), frame_idx),
                            blobs_coords
                        ])
                        all_blobs_coords.append(blobs_with_t)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError in frame {futures[future]}: {e}")
                    pbar.update(1)
    
    # Concatenate all blobs
    if all_blobs_coords:
        all_blobs_coords = np.concatenate(all_blobs_coords, axis=0)
    else:
        all_blobs_coords = np.array([])
    
    return all_blobs_coords, composite


def save_results(
    composite,
    blobs_coords,
    input_path,
    output_path,
    channel_id,
    resolution,
    algorithm,
    threshold,
    min_sigma,
    max_sigma,
    area_max,
    timepoint,
    z_min,
    z_max,
):
    """Save detection results."""
    print("Saving results...")
    os.makedirs(output_path, exist_ok=True)

    # Save composite image
    output_img_path = output_path / f"C{channel_id}_detection_img.tif"
    tifffile.imwrite(
        output_img_path,
        composite,
        imagej=True,
        metadata={"axes": "TZCYX" if timepoint is None else "ZCYX"},
        compression="zlib",
    )

    # Save coordinates
    output_coords_path = output_path / f"C{channel_id}_centers.csv"
    columns_name = (
        ["T", "Z", "Y", "X", "R"] if timepoint is None else ["Z", "Y", "X", "R"]
    )

    blob_coord_df = pd.DataFrame(blobs_coords, columns=columns_name)
    
    # Convert to microns
    blob_coord_df[["Zum", "Yum", "Xum"]] = (
        blob_coord_df[["Z", "Y", "X"]] * resolution
    )
    blob_coord_df["Rum"] = blob_coord_df["R"] * np.mean(resolution[1:])

    blob_coord_df.to_csv(output_coords_path, index_label="index")

    # Save parameters
    output_param_path = output_path / f"C{channel_id}_params.csv"
    pd.DataFrame([{
        "Blobs detected": len(blobs_coords),
        "Input path": str(input_path),
        "Channel": channel_id,
        "Resolution": str(resolution),
        "Timepoint": timepoint if timepoint is not None else "all",
        "Algorithm": algorithm,
        "Threshold": threshold,
        "Min_sigma": min_sigma,
        "Max_sigma": max_sigma,
        "Area max": area_max,
        "Z min": z_min if z_min is not None else "all",
        "Z max": z_max if z_max is not None else "all",
    }]).to_csv(output_param_path, index_label="index")
    
    print(f"Results saved to {output_path}")

def blob_detection(
    input_path,
    output_path,
    channel_id,
    resolution,
    algorithm,
    threshold,
    min_sigma,
    max_sigma,
    area_max=None,
    timepoint=None,
    z_min=None,
    z_max=None,
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
    utils.verify_input(vol, channel_id, timepoint, z_min, z_max)
    
    z, c, y, x = vol.shape[-4:]
    resolution_mi_per_px = np.array(resolution) / np.array([z, y, x])
    
    # Convert area_max from microns² to pixels²
    area_max_px = None
    if area_max is not None:
        area_max_px = area_max / (resolution_mi_per_px[1] * resolution_mi_per_px[2])

    composite = None
    blobs_coords = []

    # Set Z range
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else z
    z_range = slice(z_min, z_max)

    # Process 4D volume (no time dimension)
    if vol.ndim == 4:
        print("Processing single frame (4D volume)...")
        blobs_coords, composite = frame_blob_detection(
            vol[z_range, channel_id], algorithm, threshold, 
             min_sigma, max_sigma, area_max_px
        )
        timepoint = 0

    # Process 5D volume (with time dimension)
    elif vol.ndim == 5:
        t = vol.shape[0]
        vol_c = vol[:, z_range, channel_id]

        # Single timepoint
        if timepoint is not None:
            print(f"Processing single timepoint T={timepoint}...")
            blobs_coords, composite = frame_blob_detection(
                vol_c[timepoint], algorithm, threshold, 
                 min_sigma, max_sigma, area_max_px
            )

        # All timepoints (PARALLEL)
        else:
            blobs_coords, composite = process_frames_parallel(
                vol_c,
                algorithm,
                threshold,
                min_sigma,
                max_sigma,
                area_max_px,
                max_workers=max_workers,
            )

    # Check results
    if len(blobs_coords) == 0:
        print("No blobs detected!")
        return

    print(f"Detected {len(blobs_coords)} blobs")

    # Save results
    save_results(
        composite,
        blobs_coords,
        input_path,
        output_path,
        channel_id,
        resolution_mi_per_px,
        algorithm,
        threshold,
        min_sigma,
        max_sigma,
        area_max,
        timepoint,
        z_min,
        z_max,
    )

    elapsed = time.time() - start_time
    print(f"Detection complete")
    print(f"Total time: {elapsed:.2f}s")
    if timepoint is None and vol.ndim == 5:
        print(f"  Average: {elapsed/vol.shape[0]:.2f}s per frame")


if __name__ == "__main__":
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
    parser.add_argument("--output_path", required=True, type=pathlib.Path,
                       help="Output directory")
    parser.add_argument("--channel_id", required=True, type=int,
                       help="Channel to process")
    parser.add_argument(
        "--resolution",
        required=True,
        type=float,
        nargs=3,
        metavar=("DZ", "DY", "DX"),
        help="Voxel size in microns (depth, height, width)",
    )
    parser.add_argument("--algorithm", required=True, choices=['dog', 'log'],
                       help="Detection algorithm")
    parser.add_argument("--threshold", required=True, type=float,
                       help="Detection threshold")
    parser.add_argument("--min_sigma", type=float, default=1.0,
                       help="Minimum sigma (default: 1.0)")
    parser.add_argument("--max_sigma", type=float, default=3.0,
                       help="Maximum sigma (default: 3.0)")
    parser.add_argument("--area_max", type=float,
                       help="Maximum area in µm² for filtering")
    parser.add_argument("--timepoint", type=int,
                       help="Process single timepoint")
    parser.add_argument("--z_min", type=int,
                       help="Minimum Z slice")
    parser.add_argument("--z_max", type=int,
                       help="Maximum Z slice")
    parser.add_argument("--max_workers", type=int,
                       help="Number of parallel workers (default: CPU count - 1)")
    
    args = parser.parse_args()

    try:
        blob_detection(
            args.input_path,
            args.output_path,
            args.channel_id,
            args.resolution,
            args.algorithm,
            args.threshold,
            args.min_sigma,
            args.max_sigma,
            args.area_max,
            args.timepoint,
            args.z_min,
            args.z_max,
            args.max_workers,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
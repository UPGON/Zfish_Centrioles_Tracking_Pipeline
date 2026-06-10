import sys
import pathlib
from pathlib import Path
import os
import argparse
import time
import tifffile
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from skimage.exposure import match_histograms
from skimage.transform import rescale
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization
from utils import utils
from pairing import pairing


def merging_centers(blob_centers,merging_distance, resolution):
    new_centers = blob_centers
    centers_um = blob_centers[:, :3] * resolution
    merged = True
    while merged:
        merged = False
        tree = cKDTree(centers_um)
        
        # Find all pairs within merging_distance
        pairs = tree.query_pairs(r=merging_distance, output_type='ndarray')  # shape (M, 2)
        
        if len(pairs) == 0:
            break
        
        unmerged = np.zeros(len(centers_um), dtype=bool)
        merge_map = {}  # idx -> merged result

        for i, j in pairs:
            if unmerged[i] or unmerged[j]:
                continue
            # Merge i and j
            merge_map[i] = (new_centers[i] + new_centers[j]) / 2
            unmerged[i] = True
            unmerged[j] = True
            merged = True

        # Keep unmerged spots + new merged spots
        unmerged_centers = new_centers[~unmerged]
        new_merged = np.array(list(merge_map.values()))
        new_centers = np.concatenate([unmerged_centers, new_merged]) if len(new_merged) > 0 else unmerged_centers
        centers_um = new_centers[:, :3] * resolution

    nb_merged_centers = len(blob_centers) - len(new_centers)

    return new_centers, nb_merged_centers

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
    keep_mask = np.ones(len(blob_center), dtype=bool)

    for z in np.unique(blob_center[:, 0]).astype(int):
        thresh_yen = threshold_yen(frame[z])
        large_obj = remove_small_objects(frame[z] > thresh_yen, min_size=area_max)

        z_mask = blob_center[:, 0] == z
        ys = blob_center[z_mask, 1].astype(int)
        xs = blob_center[z_mask, 2].astype(int)

        # Clip to valid bounds
        valid = (ys >= 0) & (ys < frame.shape[1]) & (xs >= 0) & (xs < frame.shape[2])
        in_large = np.zeros(z_mask.sum(), dtype=bool)
        in_large[valid] = large_obj[ys[valid], xs[valid]]

        keep_mask[z_mask] = ~in_large

    return blob_center[keep_mask]

def normalize_to_reference(stack, ref_t=0):
    """
    Match every timepoint histogram to the reference frame.
    Works in 3D — operates on the full ZYX volume.
    """
    reference = stack[ref_t]
    stack_norm = np.zeros_like(stack, dtype=float)
    
    for t in range(len(stack)):
        stack_norm[t] = match_histograms(
            stack[t].astype(float),
            reference.astype(float)
        )
    return stack_norm

def rescale_to_isotropie(frame, z_rescale):
    # TODO: need to determine if I keep this or not
    return rescale(
        frame,
        scale= [z_rescale,1,1],  
        order = 1,             
        channel_axis  = None
    )

def frame_composite_creation(frame,frame_bf,blobs_centers,annotation, max_radius = None):
    if len(blobs_centers)==0:
        return
    if max_radius is None:  
        max_radius = (blobs_centers[:, -1].max()) * 1.5
    circle_mask = visualization.create_circle_mask(blobs_centers[:, :3], frame, radius = [max_radius])
   
    if annotation:
        mask = np.zeros(frame.shape, dtype=frame.dtype)
        annotation_mask = visualization.add_texts(mask, texts=np.arange(len(blobs_centers)).astype(str), coords=blobs_centers[:, :3])
        composite = np.stack([frame, circle_mask,annotation_mask,frame_bf], axis=1)

    else:
        composite = np.stack([frame, circle_mask,frame_bf], axis=1)
    return composite


def create_frame_composite_task(args):
    frame_idx, frame, frame_bf,blobs_centers,annotation,radius = args
    composite = frame_composite_creation(frame, frame_bf, blobs_centers,annotation,radius)
    return frame_idx, composite

def paralell_composite_creation(
    vol_c,
    vol_bf,
    blobs_center,
    annotation,
    draw_drawn_radius,
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
    if draw_drawn_radius is None:
        draw_drawn_radius = (blobs_center[:,-1].mean()) * 1.2
    
    # Prepare tasks
    tasks = [
        (ti, vol_c[ti],vol_bf[ti],blobs_center[blobs_center[:,0] == ti][:,1:], annotation,draw_drawn_radius)
        for ti in range(t)
    ]
    
    # Pre-allocate composite
    channels_nb = 4 if annotation else 3
    composite = np.empty((t, z, channels_nb, y, x), dtype=np.uint8)
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(create_frame_composite_task, task): task[0] 
                  for task in tasks}
        
        with tqdm(total=t, desc="Creating composite image", unit="frame") as pbar:
            for future in as_completed(futures):
                try:
                    frame_idx, comp = future.result()
                    
                    # Store composite
                    if comp is not None:
                        composite[frame_idx] = comp
       
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError in frame {futures[future]}: {e}")
                    pbar.update(1)
    
    return composite

def frame_blob_detection(frame, algorithm, threshold, min_sigma=1, max_sigma=5, area_max=None):
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

    # Filter by area
    if area_max:
        blobs_center = remove_large_area_element(frame, blobs_center, area_max)

    # Convert sigma to radius
    blobs_center[:, -1] = blobs_center[:, -1] * np.sqrt(3)
        
    return blobs_center
        

def frame_blob_detection_task(args):
    frame_idx, frame, algorithm, threshold, min_sigma, max_sigma, area_max = args
    blobs_center = frame_blob_detection(frame, algorithm, threshold, min_sigma, max_sigma, area_max)
    return frame_idx, blobs_center


def paralell_blob_detection(vol_c, algorithm, threshold, min_sigma, max_sigma, area_max, max_workers=None):
    t = vol_c.shape[0]
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, t)

    tasks = [
        (ti, vol_c[ti], algorithm, threshold, min_sigma, max_sigma, area_max)
        for ti in range(t)
    ]

    all_blobs_coords = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(frame_blob_detection_task, tasks),
            total=t, desc="Detecting blobs"
        ))

    for frame_idx, blobs_coords in results:
        if len(blobs_coords) > 0:
            all_blobs_coords = np.concatenate([
                np.column_stack([np.full(len(b), ti), b])
                for ti, b in results if len(b) > 0
            ], axis=0)

    return np.concatenate(all_blobs_coords, axis=0) if all_blobs_coords else np.array([])


def save_results(
    composite,
    blobs_coords,
    input_path,
    output_path,
    channel_id,
    resolution,
    algorithm,
    threshold,
    nb_merged_centers,
    min_sigma,
    max_sigma,
    area_max,
    timepoint,
    z_min,
    z_max,
):
    """Save detection results."""
    print("Saving results...")
    if output_path is None:
        output_path = Path(f"c{channel_id}_results")
    os.makedirs(output_path, exist_ok=True)

    # Save composite image
    output_img_path = output_path / f"c{channel_id}_detection_img.tif"
    tifffile.imwrite(
        output_img_path,
        composite,
        imagej=True,
        metadata={"axes": "TZCYX" if timepoint is None else "ZCYX"},
        compression="zlib",
    )

    # Save coordinates
    output_coords_path = output_path / f"c{channel_id}_centers.csv"
    columns_name = (
        ["T", "Z", "Y", "X", "R"] if timepoint is None else ["Z", "Y", "X", "R"]
    )

    blob_coord_df = pd.DataFrame(blobs_coords, columns=columns_name)
    
    #For now it's always false, let's see in the future if I want to make it a parameter
    convert_to_micron = False
    if convert_to_micron:
        # Convert to microns
        blob_coord_df[["Zum", "Yum", "Xum"]] = (
            blob_coord_df[["Z", "Y", "X"]] * resolution
        )
        blob_coord_df["Rum"] = blob_coord_df["R"] * np.mean(resolution[1:])

    blob_coord_df.to_csv(output_coords_path, index_label="index")

    # Save parameters
    output_param_path = output_path / f"c{channel_id}_params.csv"
    pd.DataFrame([{
        "Blobs detected": len(blobs_coords),
        "Input path": str(input_path),
        "Channel": channel_id,
        "Resolution [px/um]": str(resolution),
        "Timepoint": timepoint if timepoint is not None else "all",
        "Algorithm": algorithm,
        "Threshold": threshold,
        "Min_sigma": min_sigma,
        "Max_sigma": max_sigma,
        "Nb merged centers":nb_merged_centers,
        "Area max": area_max,
        "Z min": z_min if z_min is not None else "all",
        "Z max": z_max if z_max is not None else "all",
    }]).to_csv(output_param_path, index_label="index")
    
    print(f"Results saved to {output_path}")

def blob_detection(
    input_path,
    channel_id,
    algorithm,
    threshold,
    min_radius,
    max_radius,
    output_path = None,
    area_max=None,
    merging_distance=None,
    timepoint=None,
    z_min=None,
    z_max=None,
    annotation = True,
    drawn_radius = None,
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
    # Get pixel size: [px/um]
    pixel_size = utils.get_pixel_size(input_path)
    
    # Convert area_max from microns² to pixels²
    area_max_px = None
    if area_max is not None:
        area_max_px = area_max / (pixel_size[1] * pixel_size[2])

    min_sigma = min_radius/ (pixel_size[1:].mean() * np.sqrt(3))
    max_sigma = max_radius /(pixel_size[1:].mean() * np.sqrt(3))

    print(f"Sigma parameters: ")
    print(f"    Min sigma: {min_sigma:.2f}: ")
    print(f"    Max sigma: {max_sigma:.2f}: ")

    composite = None
    blobs_coords = []
    nb_merged_centers = 0

    # Set Z range
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else z
    z_range = slice(z_min, z_max)

    # Process 4D volume (no time dimension)
    if vol.ndim == 4:
        print("Processing single frame (4D volume)...")
        vol_c = vol[z_range, py_channel_id]
        vol_bf = vol[z_range, -1]
        blobs_coords = frame_blob_detection(
            vol_c, algorithm, threshold, 
             min_sigma, max_sigma, area_max_px
        )
        if merging_distance:
            print("Merging centers")
            blobs_coords,nb_merged_centers = merging_centers(blobs_coords, merging_distance,pixel_size)
        print("Creating composite")
        composite = frame_composite_creation(
                vol_c, vol_bf,blobs_coords,annotation,drawn_radius
        )
        timepoint = 0

    # Process 5D volume (with time dimension)
    elif vol.ndim == 5:
        t = vol.shape[0]
        vol_c = vol[:, z_range, py_channel_id]
        vol_bf = vol[:, z_range, -1]

        # Single timepoint
        if timepoint is not None:
            print(f"Processing single timepoint T={timepoint}...")
            blobs_coords = frame_blob_detection(
                vol_c[timepoint], algorithm, threshold, 
                 min_sigma, max_sigma, area_max_px
            )
            if merging_distance:
                blobs_coords,nb_merged_centers = merging_centers(blobs_coords, merging_distance,pixel_size)
            print("Creating composite")
            composite = frame_composite_creation(
                vol_c[timepoint], vol_bf[timepoint],blobs_coords,annotation, drawn_radius
            )

        # All timepoints (PARALLEL)
        else:
            blobs_coords = paralell_blob_detection(
                vol_c,
                algorithm,
                threshold,
                min_sigma,
                max_sigma,
                area_max_px,
                max_workers=max_workers,
            )
            if merging_distance:
                blobs_coords,nb_merged_centers = merging_centers(blobs_coords, merging_distance,pixel_size)
            composite = paralell_composite_creation(
                vol_c,
                vol_bf,
                blobs_coords,
                annotation,
                drawn_radius,
                max_workers
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
        pixel_size,
        algorithm,
        threshold,
        nb_merged_centers,
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
    parser.add_argument("--algorithm", required=True, choices=['dog', 'log'],
                       help="Detection algorithm")
    parser.add_argument("--threshold", required=True, type=float,
                       help="Detection threshold")
    parser.add_argument("--output_path", type=pathlib.Path,
                       help="Output directory")
    parser.add_argument("--min_radius", type=float, default=0.2,
                       help="Minimum radius [um] (default: 0.2)")
    parser.add_argument("--max_radius", type=float, default=1,
                       help="Maximum radius [um] (default: 1.3)")
    parser.add_argument("--area_max", type=float,
                       help="Maximum area in µm² for filtering")
    parser.add_argument("--merging_distance", type=float,
                       help="Maximum area in µm² for filtering")
    parser.add_argument("--timepoint", type=int,
                       help="Process single timepoint")
    parser.add_argument("--z_min", type=int,
                       help="Minimum Z slice")
    parser.add_argument("--z_max", type=int,
                       help="Maximum Z slice")
    parser.add_argument("--annotation", type=bool, default=True,
                       help="If the idx label should be in the results images")
    parser.add_argument("--drawn_radius", type=float, default=None,
                       help="If the idx label should be in the results images")
    parser.add_argument("--max_workers", type=int,
                       help="Number of parallel workers (default: CPU count - 1)")
    
    
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        blob_detection(
            args.input_path,
            args.channel_id,
            args.algorithm,
            args.threshold,
            args.min_radius,
            args.max_radius,
            args.output_path,
            args.area_max,
            args.merging_distance,
            args.timepoint,
            args.z_min,
            args.z_max,
            args.annotation,
            args.drawn_radius,
            args.max_workers,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
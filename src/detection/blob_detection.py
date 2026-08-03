import sys
import pathlib
from pathlib import Path
import os
import argparse
import time
import tifffile
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from skimage.exposure import match_histograms
from skimage.transform import rescale
from scipy.spatial import cKDTree


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization
from utils import utils
import constants

def merging_centers(dict_centers, merging_distance, resolution):
    """Merge nearby centers within merging_distance.it se
    
    Args:
        dict_centers: Dict with 'coords' (N,3) and 'areas' (N,)
        merging_distance: Distance threshold in microns
        resolution: Voxel size [dz, dy, dx]
    
    Returns:
        Tuple of (merged_dict_centers, nb_merged_centers)
    """
    coords = dict_centers["coords"].copy()
    areas = dict_centers["areas"].copy()
    intensities = dict_centers["intensities"].copy()
    if len(coords)!= len(areas) or len(coords) != len(intensities):
        raise ValueError("All mesure features should have the same size")

    temporal_data = "T" in dict_centers.keys()
    if temporal_data:
        frames = dict_centers["T"].copy()
    
    nb_original = len(coords)
    merged = True
    
    while merged:
        merged = False
        centers_um = coords * resolution
        tree = cKDTree(centers_um)
        
        # Find all pairs within merging_distance
        pairs = tree.query_pairs(r=merging_distance, output_type='ndarray')
        
        if len(pairs) == 0:
            break
        
        unmerged_mask = np.ones(len(coords), dtype=bool)
        new_coords_list = []
        new_areas_list = []
        new_intensities_list = []
        if temporal_data:
            new_frames_list = []

        for i, j in pairs:
            if not unmerged_mask[i] or not unmerged_mask[j]:
                continue
            
            # Merge i and j
            merged_coords = (coords[i] + coords[j]) / 2
            merged_area = areas[i] + areas[j] if areas[i]!= areas[j] else areas[i]
            merged_intensities = (intensities[i] + intensities[j])/2
            if temporal_data:
                merged_frames = (frames[i] + frames[j]) // 2

            new_coords_list.append(merged_coords)
            new_areas_list.append(merged_area)
            new_intensities_list.append(merged_intensities)
            if temporal_data:
                new_frames_list.append(merged_frames)
            
            unmerged_mask[i] = False
            unmerged_mask[j] = False
            merged = True
        
        # Keep unmerged spots + new merged spots
        unmerged_coords = coords[unmerged_mask]
        unmerged_areas = areas[unmerged_mask]
        unmerged_intensities = intensities[unmerged_mask]
        if temporal_data:
            unmerged_frames = frames[unmerged_mask]
        
        if new_coords_list:
            coords = np.vstack([unmerged_coords, np.array(new_coords_list)])
            areas = np.concatenate([unmerged_areas, np.array(new_areas_list)])
            intensities = np.concatenate([unmerged_intensities, np.array(new_intensities_list)])
            if temporal_data:
                frames = np.concatenate([unmerged_frames, np.array(new_frames_list)])

        else:
            coords = unmerged_coords
            areas = unmerged_areas
            intensities  = unmerged_intensities
            if temporal_data:
                frames = unmerged_frames
    
    nb_merged_centers = nb_original - len(coords)

    if temporal_data:
        new_dict_centers = {"coords": coords, "T": frames, "areas": areas, "intensities": intensities}
    else:
        new_dict_centers = {"coords": coords, "areas": areas, "intensities": intensities}
    
    return new_dict_centers , nb_merged_centers

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
        max_radius = (np.sqrt(blobs_centers["areas"].max() / np.pi)) * 1.5
    circle_mask = visualization.create_circle_mask(blobs_centers["coords"][:, :3], frame, radius = [max_radius])
   
    if annotation:
        mask = np.zeros(frame.shape, dtype=frame.dtype)
        annotation_mask = visualization.add_texts(mask, texts=np.arange(len(blobs_centers["coords"])).astype(str), coords=blobs_centers["coords"][:, :3])
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
        draw_drawn_radius = (blobs_center["T"].mean()) * 1.2
    
    # Prepare tasks
    tasks = []
    for ti in range(t):
        frame_mask = blobs_center["T"] == ti

        if np.any(frame_mask):
            frame_blobs = {
                "coords": blobs_center["coords"][frame_mask],
                "areas": blobs_center["areas"][frame_mask],
                "intensities": blobs_center["intensities"][frame_mask],
            }
        else:
            frame_blobs = {
                "coords": np.empty((0, 3), dtype=blobs_center["coords"].dtype),
                "areas": np.empty((0,), dtype=blobs_center["areas"].dtype),
                "intensities": np.empty((0,), dtype=blobs_center["intensities"].dtype),
            }

        tasks.append((ti, vol_c[ti], vol_bf[ti], frame_blobs, annotation, draw_drawn_radius))
   
    
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

def remove_out_of_field(centers,frame, gap_limit =1):
    z,y,x = frame.shape
    valid_mask = [center[1] > gap_limit and center[1] < (y-gap_limit) and center[2] > gap_limit and center[2] < (x-gap_limit) for center in centers]
    return centers[valid_mask]

def remove_out_of_border(dict_centers, frame, window_size=40, zero_frac_thresh=0.2):
    """Remove centers adjacent to large black background borders.

    Args:
        dict_centers: Dict with keys 'coords' (N,3), 'areas' (N,), 'intensities' (N,), and optional 'T'
        frame: 3D image (Z, Y, X)
        window_size: Size of window around each center
        zero_frac_thresh: Threshold for fraction of zero pixels in window

    Returns:
        Filtered dict_centers with same structure but only valid centers.
    """
    if len(dict_centers["coords"]) == 0:
        return dict_centers

    coords = dict_centers["coords"]
    areas = dict_centers["areas"]
    intensities = dict_centers["intensities"]
    has_temporal = "T" in dict_centers
    if has_temporal:
        frames = dict_centers["T"]

    valid_mask = []

    for i in range(len(coords)):
        center = coords[i]
        frameT = frames[i] if has_temporal else None
        cz, cy, cx = int(center[0]), int(center[1]), int(center[2])

        window_range = utils.get_window_range(window_size, center, frame.shape[-3:])
        if has_temporal:
            window = frame[frameT][cz][window_range[1:3]]
        else:
            window = frame[cz][window_range[1:3]]

        zero_frac = np.mean(window == 0)

        # If too many pixels are zero, consider it touching a black crop border
        if zero_frac > zero_frac_thresh:
            valid_mask.append(False)
        else:
            valid_mask.append(True)

    valid_mask = np.array(valid_mask)
    
    # Build filtered dict
    filtered_dict = {
        "coords": coords[valid_mask],
        "areas": areas[valid_mask],
        "intensities": intensities[valid_mask],
    }
    if has_temporal:
        filtered_dict["T"] = frames[valid_mask]

    return filtered_dict

def measure_features(centers,frame, window_size = 40):
    areas = np.empty(len(centers))
    intensities = np.empty(len(centers))

    for i,center in enumerate(centers):
        z,y,x = center.astype(int)
        window_range = utils.get_window_range(window_size, [y,x], frame.shape[1:3])
        wz,wy,wx = utils.get_windowed_coord(window_size, center,frame.shape).astype(int)

        img = frame[z][window_range]
        gaussian_img = ndimage.gaussian_filter(img,sigma=1)
        #yen_img = gaussian_img > threshold_yen(gaussian_img)
        yen_img = gaussian_img > threshold_otsu(gaussian_img)

        yen_labels = label(yen_img)
        areas[i] = ndimage.sum(yen_img, labels=yen_labels, index=yen_labels[wy,wx])
        intensities[i] = gaussian_img[wy,wx]

    return areas, intensities


def frame_blob_detection(frame, algorithm, threshold, min_sigma=1, max_sigma=5, window_size=40, area_max=None):
    """Detect blobs in a single frame.
    
    Args:
        frame: 3D image (Z, Y, X)
        algorithm: 'dog' or 'log'
        threshold: Detection threshold
        min_sigma: Minimum sigma
        max_sigma: Maximum sigma
        area_max: Maximum area threshold
    
    Returns:
        Dict with 'coords' (N,3) and 'areas' (N,)
    """
    # Detect blobs
    blobs_center = apply_detection_algorithm(
        frame, algorithm, min_sigma, max_sigma, threshold
    )[:,:-1] # We don't want the estimated radius by the algo, we'll rather compute our own estimation

    blobs_center = remove_out_of_field(blobs_center, frame)

    areas, intensities = measure_features(blobs_center, frame,window_size)

    # Filter by area if specified
    if area_max is not None:
        valid_mask = areas <= area_max
        blobs_center = blobs_center[valid_mask]
        areas = areas[valid_mask]
        intensities = intensities[valid_mask]

    dict_centers = {"coords": blobs_center, "areas": areas,"intensities":intensities}
  
    return dict_centers
        

def frame_blob_detection_task(args):
    frame_idx, frame, algorithm, threshold, min_sigma, max_sigma,window_size, area_max = args
    blobs_center = frame_blob_detection(frame, algorithm, threshold, min_sigma,max_sigma,window_size, area_max)
    return frame_idx, blobs_center


def paralell_blob_detection(vol_c, algorithm, threshold, min_sigma, max_sigma,window_size, area_max, max_workers=None):
    t = vol_c.shape[0]
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, t)

    tasks = [
        (ti, vol_c[ti], algorithm, threshold, min_sigma, max_sigma,window_size, area_max)
        for ti in range(t)
    ]

    all_blobs_coords = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(frame_blob_detection_task, tasks),
            total=t, desc="Detecting blobs"
        ))

    for frame_idx, blobs_coords in results:
        if len(blobs_coords["coords"]) > 0:
            all_blobs_coords.append({
                "coords":    blobs_coords["coords"],
                "areas":     blobs_coords["areas"],
                "intensities": blobs_coords["intensities"],
                "T":         np.full(len(blobs_coords["coords"]), frame_idx),
            })

    if all_blobs_coords:
        return {
            "coords": np.concatenate([blob["coords"] for blob in all_blobs_coords], axis=0),
            "areas": np.concatenate([blob["areas"] for blob in all_blobs_coords], axis=0),
            "intensities": np.concatenate([blob["intensities"] for blob in all_blobs_coords], axis=0),
            "T": np.concatenate([blob["T"] for blob in all_blobs_coords], axis=0),
        }
    else:
        return {"coords": np.array([]).reshape(0, 3), "areas": np.array([]),"intensities":np.array([]), "T": np.array([])}



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
    
    # Extract data from dict_centers
    coords = blobs_coords["coords"]
    areas = blobs_coords["areas"]
    intensities = blobs_coords["intensities"]
    
    # Build output array based on timepoint
    if "T" in blobs_coords:
        output_data = np.column_stack([blobs_coords["T"], coords, areas, intensities])
        columns_name = ["T", "Z", "Y", "X", "Area","Intensity"]
    else:
        output_data = np.column_stack([coords, areas, intensities])
        columns_name = ["Z", "Y", "X", "Area","Intensity"]
    
    blob_coord_df = pd.DataFrame(output_data, columns=columns_name)
    
    #For now it's always false, let's see in the future if I want to make it a parameter
    convert_to_micron = False
    if convert_to_micron:
        # Convert to microns
        blob_coord_df[["Zum", "Yum", "Xum"]] = (
            blob_coord_df[["Z", "Y", "X"]] * resolution
        )
        blob_coord_df["Area_um2"] = blob_coord_df["Area"] * (resolution[1] * resolution[2])

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
        area_max_px = area_max / (pixel_size[1])
    window_size_px = int(constants.ROI_WINDOW_SIZE / pixel_size[1])
    print(f"Window_size {window_size_px}")

    min_sigma = min_radius/ (pixel_size[1:].mean() * np.sqrt(3))
    max_sigma = max_radius /(pixel_size[1:].mean() * np.sqrt(3))

    print(f"Sigma parameters: ")
    print(f"    Min sigma: {min_sigma:.2f}: ")
    print(f"    Max sigma: {max_sigma:.2f}: ")

    composite = None
    dict_centers = {}
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
        dict_centers = frame_blob_detection(
            vol_c, algorithm, threshold, 
             min_sigma, max_sigma,window_size_px, area_max_px
        )
        dict_centers = remove_out_of_border(dict_centers, vol_bf, window_size_px)
        if merging_distance:
            print("Merging centers")
            dict_centers,nb_merged_centers = merging_centers(dict_centers, merging_distance, pixel_size)
        print("Creating composite")
        composite = frame_composite_creation(
                vol_c, vol_bf,dict_centers,annotation,drawn_radius
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
            dict_centers = frame_blob_detection(
                vol_c[timepoint], algorithm, threshold, 
                 min_sigma, max_sigma,window_size_px, area_max_px
            )
            dict_centers = remove_out_of_border(dict_centers, vol_bf[timepoint], window_size_px)
            if merging_distance:
                dict_centers,nb_merged_centers = merging_centers(dict_centers, merging_distance,pixel_size)
            print("Creating composite")
            composite = frame_composite_creation(
                vol_c[timepoint], vol_bf[timepoint],dict_centers,annotation, drawn_radius
            )

        # All timepoints (PARALLEL)
        else:
            dict_centers = paralell_blob_detection(
                vol_c,
                algorithm,
                threshold,
                min_sigma,
                max_sigma,
                window_size_px,
                area_max_px,
                max_workers=max_workers,
            )
            dict_centers = remove_out_of_border(dict_centers, vol_bf, window_size_px)
            if merging_distance:
                dict_centers,nb_merged_centers = merging_centers(dict_centers, merging_distance,pixel_size)
            print("Creating composite")
            composite = paralell_composite_creation(
                vol_c,
                vol_bf,
                dict_centers,
                annotation,
                drawn_radius,
                max_workers
            )


    # Check results
    if len(dict_centers["coords"]) == 0:
        print("No blobs detected!")
        return

    print(f"Detected {len(dict_centers['coords'])} blobs")

    # Save results
    save_results(
        composite,
        dict_centers,
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
    parser.add_argument("--drawn_radius", type=float,
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
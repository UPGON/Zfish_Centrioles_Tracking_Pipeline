import argparse
import pathlib
import time
import tifffile
import numpy as np
from itertools import product
from stardist.models import StarDist3D
from csbdeep.utils import Path, normalize
import tqdm
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
import os
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils
from visualization import visualization


def normalizes_frame(frame, pmin=1, pmax=99.8):
    """Normalize a 3D image frame using percentile-based normalization.
    Args:
        frame: 3D image (Z, Y, X)
        pmin: Minimum percentile for normalization
        pmax: Maximum percentile for normalization
    Returns:
        Normalized 3D image
    """
    axis_norm = np.arange(len(frame.shape))
    norm_frame = normalize(frame, pmin=pmin, pmax=pmax, axis=axis_norm)
    return norm_frame

def save_results(labels, polys, input_path, output_path, channel_id, timepoint,z_min,z_max, model_path):
    os.makedirs(output_path, exist_ok=True)

    # Save the annotated image
    output_img_path = output_path / f"c{channel_id}_segmentation.tif"
    tifffile.imwrite(output_img_path, labels, imagej=True, compression="zlib")

    # Save the blobs center coordinate

    polys_df = pd.DataFrame(polys["points"], columns=["Z","Y","X"])
    if timepoint is None:
        polys_df["T"] = polys["timeframe"]
    output_coords_path = output_path / f"c{channel_id}_centers.csv"
    polys_df.to_csv(output_coords_path, index_label="index")

    # Save the parameters used for the detection
    output_param_path = output_path / f"c{channel_id}_params.csv"
    pd.DataFrame(
        [
            {
                "Input path": str(input_path),
                "Channel": channel_id,
                "Timepoint": timepoint if timepoint is not None else "all",
                "Zmin": z_min,
                "Zmax": z_max,
                "Model used": model_path if model_path is not None else "3D_demo",
                "Blobs detected": len(polys["points"]),
            }
        ]
    ).to_csv(output_param_path, index_label="index")

def frame_composite_creation(frame,centers, max_radius = 4):
    if len(centers)==0:
        return
    circle_mask = visualization.create_circle_mask(centers, frame, radius = [max_radius], thickness=2)
   
    mask = np.zeros(frame.shape, dtype=frame.dtype)
    annotation_mask = visualization.add_texts(mask, texts=np.arange(len(centers)).astype(str), coords=centers, fontScale = 0.6)
    return circle_mask, annotation_mask

def segment_frame(frame, model_path=None, proba_thresh=None, nms_thresh=None, scale=None):
    """Segment a 3D image using the StarDist model and save the results.
    Args:
        img (numpy.ndarray): The input image to segment.
        output_path (str): The path where the segmented image will be saved.
        model_name (str): The name of the pre-trained StarDist model to use for segmentation.
    Returns:
        None: The function saves the segmented image to the specified output path.
    """
    if model_path is None:
        model = StarDist3D.from_pretrained("3D_demo")
    else:
        model = StarDist3D(None, model_path)
    labels, polys = model.predict_instances(
        frame,  # The image must be normalized
        axes="ZYX",
        prob_thresh=proba_thresh,  # Detection probability threshold
        nms_thresh=nms_thresh,  # Remove detections overlapping by more than this threshold
        scale=scale,  # Higher values are suitable for lower resolution data
        n_tiles  =model._guess_n_tiles(frame),
        return_labels=True,
    )
    return [labels.astype(np.uint), polys]


def frame_segmentation_task(args):
    frame_idx, frame, model_path, proba_thresh, nms_thresh, scale = args
    norm_frame = normalizes_frame(frame)

    labels, polys = segment_frame(norm_frame,model_path,proba_thresh,nms_thresh,scale)
    return frame_idx, labels.astype(np.uint), polys


def process_segmentation_parallel(
    vol_c,
    model_path,
    proba_thresh = None,
    nms_thresh = None,
    scale = None,
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
    
    # Load the model
    if model_path is None:
        model = StarDist3D.from_pretrained("3D_demo")
    else:
        model = StarDist3D(None, model_path)

    # Prepare tasks
    tasks = [
        (ti, vol_c[ti], model_path, proba_thresh, nms_thresh, scale)
        for ti in range(t)
    ]
    
    # Pre-allocate composite
    all_labels = np.empty((t, z, y, x), dtype=np.uint8)
    all_polys = []
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(frame_segmentation_task, task): task[0] 
                  for task in tasks}
        
        with tqdm(total=t, desc="Detecting blobs", unit="frame") as pbar:
            for future in as_completed(futures):
                try:
                    frame_idx, labels, polys = future.result()
                    
                    # Store composite
                    if labels is not None:
                        all_labels[frame_idx] = labels
                    
                    # Add timepoint column to blobs
                    if polys is not None and len(polys["points"] > 0):
                        all_polys.append({
                            "points": polys["points"].copy(),
                            "timeframe": np.full(len(polys["points"]),frame_idx)
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError in frame {futures[future]}: {e}")
                    pbar.update(1)
    
    # Concatenate all blobs
    if all_polys:
        all_polys = {
            "points": np.concatenate([p["points"] for p  in all_polys], axis =0),
            "timeframe": np.concatenate([p["timeframe"] for p  in all_polys], axis =0)
        }
    else:
        all_polys = {}
    
    return all_labels, all_polys


def segmentation(input_path, output_path, channel_id, model_resolution, model_path=None,proba_thresh=None, nms_thresh=None,timepoint=None, z_min=None, z_max=None,resolution_as_scale=False):
    """Main segmentation pipeline.

    Args:
        input_path: Path to input TIFF
        output_path: Output directory
        channel_id: Channel to process
        timepoint: Optional specific timepoint to process
        model_name: Name of the pre-trained StarDist model to use for segmentation

    Returns:
        None: The function saves the segmented image and coordinates to the specified output path.
    """
    vol = tifffile.memmap(input_path)
    py_channel_idx = channel_id -1
    utils.verify_input(vol, py_channel_idx, timepoint, z_min, z_max)
    z, c, y, x = vol.shape[-4:]

    # If no z min and max were specified, then we process over the whole z stack
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else z
    z_range = range(z_min, z_max)

    resolution = utils.get_pixel_size(input_path)
    print(f"Image resolution {resolution} [px/um]")
    if resolution_as_scale:
        scale = model_resolution
    else:
        scale = resolution/model_resolution
    print(f"Scale used in the model {scale} px/um")

    vol_labels = None
    vol_polys = {}

     # Process 4D volume (no time dimension)
    if vol.ndim == 4:
        timepoint = 0
        print("Processing single frame (4D volume)...")
        norm_img = normalizes_frame(vol[z_range, py_channel_idx])

        [vol_labels, vol_polys] = segment_frame(norm_img, model_path,scale=scale,proba_thresh=proba_thresh,nms_thresh=nms_thresh)
        circle_mask, annotation_mask = frame_composite_creation(vol[z_range, py_channel_idx],vol_polys["points"])
        composite = np.stack([vol[z_range, py_channel_idx],vol_labels,circle_mask, annotation_mask],axis=1)

    # Process 5D volume (with time dimension)
    elif vol.ndim == 5:
        t = vol.shape[0]
        vol_c = vol[:, z_range, py_channel_idx]

        # Single timepoint
        if timepoint is not None:
            print(f"Processing single timepoint T={timepoint}...")
            norm_img = normalizes_frame(vol_c[timepoint])

            [vol_labels, vol_polys] = segment_frame(norm_img, model_path,scale=scale,proba_thresh=proba_thresh,nms_thresh=nms_thresh)
            circle_mask, annotation_mask = frame_composite_creation(vol_c[timepoint],vol_polys["points"])
            composite = np.stack([vol_c[timepoint],vol_labels,circle_mask, annotation_mask],axis=1)

        # All timepoints (PARALLEL)
        else:
            vol_labels, vol_polys = process_segmentation_parallel(
                vol_c,
                model_path,
                proba_thresh=proba_thresh,nms_thresh=nms_thresh,
                max_workers=4,
            )
            composite = np.stack([vol_c,vol_labels],axis=2)

    if len(vol_polys) == 0:  # avoids error if empty
        print("No spots detected")
        return
    print(f"{len(vol_polys['points'])} elements detected")
    
    composite = composite.astype(np.uint8)
    save_results(
        composite,
        vol_polys,
        input_path,
        output_path,
        channel_id,
        timepoint,
        z_min,
        z_max,
        model_path,
    )

    print("Segmentation completed")



def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Crop the given image using the provided coordinates"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--channel_id", required=True, type=int)   
    parser.add_argument(
        "--model_resolution",
        required=True,
        type=float,
        nargs=3,
        metavar=("DZ", "DY", "DX"),
        help="Resolution of the trained images of the model in [um/px]",
    )
    parser.add_argument("--model_path", type=str) 
    parser.add_argument("--proba_thresh", type=float)  
    parser.add_argument("--nms_thresh", type=float)   
    parser.add_argument("--timepoint", type=int)
    parser.add_argument("--z_min", type=int)
    parser.add_argument("--z_max", type=int)
    parser.add_argument("--resolution_as_scale", type=bool, default = False)

    return parser


if __name__ == "__main__":
    """Command-line interface for segmenting an image using a pre-trained StarDist model.
    Usage:
        python stardist_segmentation.py --input_path <path_to_input_image> --output_path <path_to_output_directory> --model_name <name_of_pretrained_model> --channel_id <channel_id> [--timepoint <timepoint_to_process>]
    Args:
        --input_path (str): The path of the image to be segmented. The image must be in 3D with format (T,Z,Y,X).
        --output_path (str): The path where the segmented image will be saved (must be a directory).
        --model_name (str, optional): The name of the pre-trained StarDist model to use for segmentation. If not provided, the default model '3D_demo' will be used.
        --channel_id (int): The channel to process.
        --timepoint (int, optional): If specified, only this timepoint will be processed. If not provided, all timepoints will be processed.
    """
    args = _build_arg_parser().parse_args()

    segmentation(
        args.input_path,
        args.output_path,
        args.channel_id,
        args.model_resolution,
        args.model_path,
        args.proba_thresh,
        args.nms_thresh,
        args.timepoint,
        args.z_min,
        args.z_max,
        args.resolution_as_scale,
    )

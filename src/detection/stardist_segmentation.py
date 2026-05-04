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

import sys

sys.path.append(r"K:\users\voland\Code\Zfish_Centrioles_Tracking_Pipeline")
from src.utils import utils


def normalizes_frame(frame, pmin=1, pmax=99.8):
    """Normalize a 3D image frame using percentile-based normalization.
    Args:
        frame: 3D image (Z, Y, X)
        pmin: Minimum percentile for normalization
        pmax: Maximum percentile for normalization
    Returns:
        Normalized 3D image
    """
    Z = frame.shape[0]
    norm_volumes = np.empty(frame.shape)
    for z in range(Z):
        slice = frame[z]
        norm_volumes[z] = normalize(slice, pmin, pmax)
    return norm_volumes


def save_results(labels, polys, input_path, output_path, channel_id, timepoint):
    # Save the annotated image
    output_img_path = output_path / f"C{channel_id}_detected_img.tif"
    tifffile.imwrite(output_img_path, labels, imagej=True, compression="zlib")

    # Save the blobs center coordinate
    polys_df = pd.DataFrame(polys)
    output_coords_path = output_path / f"C{channel_id}_detected_coords.csv"
    polys_df.to_csv(output_coords_path, index_label="index")

    # Save the parameters used for the detection
    output_param_path = output_path / f"C{channel_id}_params.csv"
    pd.DataFrame(
        [
            {
                "Input path": str(input_path),
                "Channel": channel_id,
                "Timepoint": timepoint if timepoint is not None else "all",
                "Blobs detected": len(polys),
            }
        ]
    ).to_csv(output_param_path, index_label="index")


def segment_frame(frame, model_name="3D_demo", proba_thresh=0.5, nms_thresh=0.1, scale=2):
    """Segment a 3D image using the StarDist model and save the results.
    Args:
        img (numpy.ndarray): The input image to segment.
        output_path (str): The path where the segmented image will be saved.
        model_name (str): The name of the pre-trained StarDist model to use for segmentation.
    Returns:
        None: The function saves the segmented image to the specified output path.
    """
    model = StarDist3D.from_pretrained(model_name)
    labels, polys = model.predict_instances(
        frame,  # The image must be normalized
        axes="ZYX",
        prob_thresh=proba_thresh,  # Detection probability threshold
        nms_thresh=nms_thresh,  # Remove detections overlapping by more than this threshold
        scale=scale,  # Higher values are suitable for lower resolution data
        return_labels=True,
    )
    return [labels, polys]


def segmentation(input_path, output_path, channel_id, timepoint, z_min, z_max, model_name):
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
    [t, z, y, x] = vol_c.shape
    utils.verify_input(vol, channel_id, timepoint, z_min, z_max)

    # If no z min and max were specified, then we process over the whole z stack
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else z
    z_range = range(z_min, z_max)
    vol_c = vol[:, z_range, channel_id]

    composite = None
    polys = {}

    if timepoint is not None:
        print(f"Processing a single frame at T={timepoint}")
        norm_img = normalizes_frame(vol_c[timepoint])

        [composite, polys] = segment_frame(norm_img, model_name)
    else:
        print(f"Processing all the frames")
        composite = np.empty((t, len(z_range), 2, y, x), dtype=np.uint8)

        for ti in tqdm(range(t), desc="Detecting blobs in frames", unit="frame"):
            norm_vol_c_ti = normalizes_frame(vol_c[ti])
            [composite[ti], polys_ti] = segment_frame(norm_vol_c_ti, model_name)
            if polys is not None:
                polys_ti["timeframe"] = np.full(len(polys_ti), ti)
                polys.update(polys_ti)

    if len(polys) == 0:  # avoids error if empty
        print("No spots detected")
        return

    save_results(
        composite,
        polys,
        input_path,
        output_path,
        channel_id,
        timepoint,
        z_min,
        z_max,
        model_name,
    )


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
    parser = argparse.ArgumentParser(
        description="Crop the given image using the provided coordinates"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--channel_id", required=True, type=int)
    parser.add_argument("--timepoint", type=int)
    parser.add_argument("--z_min", type=int)
    parser.add_argument("--z_max", type=int)
    parser.add_argument("--model_name", nargs="?", type=str, const="3D_demo")
    args = parser.parse_args()

    segmentation(
        args.input_path,
        args.output_path,
        args.channel_id,
        args.timepoint,
        args.z_min,
        args.z_max,
        args.model_name,
    )

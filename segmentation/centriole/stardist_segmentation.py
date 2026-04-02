import argparse
import pathlib
import time
import tifffile
import numpy as np
from itertools import product
from stardist.models import StarDist3D
from cellpose import models, core, io, plot
from csbdeep.utils import Path, normalize
import tqdm
import pandas as pd

def normalise3D(vol):
    Z= vol.shape[0]
    norm_volumes = np.empty(vol.shape, vol.dtype)
    for z in range(Z):
            img = vol[z]
            norm_volumes[z] = normalize(img, 1,99.8)
    return norm_volumes

def normalize4D(vol, pmin, pmax, axis=None):
    T, Z, C, Y, X = vol.shape
    axis_norm = (0,1,2)
    norm_volumes = np.empty(vol.shape, vol.dtype)
    for t, z, c in product(range(T), range(Z), range(C)):
        img = vol[t,z,c]
        norm_volumes[t,z,c] = normalize(img, 1,99.8, axis=axis_norm)

def normalizeVol(vol):
    vol_axis = vol.shape
    norm_axis = np.arange(len(vol_axis - 2))
    norm_volumes = np.empty(vol.shape, vol.dtype)
    norm_ranges = [np.arange(vol_axis[i]) for i in norm_axis]
    for idx in product(*norm_ranges):
        img = vol[idx]
        norm_volumes[idx] = (img - img.min()) / (img.max() - img.min())

    return norm_volumes

def save_results(labels, polys, output_path, channel_id):
    output_img = output_path / f"C{channel_id}_detected_img.tif"
    tifffile.imwrite(output_img, 
                        labels,
                        imagej=True,
                        compression='zlib'
                        )
    
    polys_df = pd.DataFrame({"dist": polys["dist"], "points": polys["points"], "prob": polys["prob"]})
    output_coords = output_path / f"C{channel_id}_detected_coords.csv"
    polys_df.to_csv(output_coords, index_label ="index")

def startdist_segm(img, output_path, model_name, channel_id):
    """ Segment the image using the StarDist model and save the results.
    Args:
        img (numpy.ndarray): The input image to segment.
        output_path (str): The path where the segmented image will be saved.
        model_name (str): The name of the pre-trained StarDist model to use for segmentation.
    Returns:
        None: The function saves the segmented image to the specified output path.
    """
    model = StarDist3D.from_pretrained(model_name)
    labels, polys = model.predict_instances(
        img,  # The image must be normalized
        axes="ZYX",
        prob_thresh=0.5,  # Detection probability threshold
        nms_thresh=0.1,  # Remove detections overlapping by more than this threshold
        scale=1,  # Higher values are suitable for lower resolution data
        return_labels=True,
    )
    save_results(labels, polys, output_path, channel_id)
    

def segmentation(input_path, output_path, model_name, channel_id, timepoint):
    vol = tifffile.memmap(input_path)

    vol_c = vol[:,:,channel_id]

    if timepoint is not None:
        print(f"Processing a single frame at T={timepoint}")
        img = vol_c[timepoint]
        norm_img = normalizeVol(img)

        startdist_segm(norm_img, output_path, model_name, channel_id)
    else:
        print(f"Processing all the frames")
        [t,z,y,x] = vol_c.shape
        composite = np.empty((t,z,2,y,x), dtype = np.uint8)

        norm_vol_c = normalize4D(vol_c)

        for ti in tqdm(range(t), desc="Detecting blobs in frames", unit="frame"):
            startdist_segm(norm_vol_c[ti], output_path, model_name, channel_id)

if __name__ == "__main__":
    """ Command-line interface for segmenting an image using a pre-trained StarDist model.
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
    parser.add_argument("--model_name", nargs='?', type=str, const='3D_demo')
    parser.add_argument("--channel_id", required=True, type = int)
    parser.add_argument("--timepoint", type = int)
    args = parser.parse_args()

    segmentation(args.input_path, args.output_path, args.model_name, args.channel_id, args.timepoint)

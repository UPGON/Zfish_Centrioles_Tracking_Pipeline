import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import requests
import yaml
import json
from pathlib import Path

def verify_input(vol, channel_id, timepoint=None, z_min=None, z_max=None):
    """Verify input parameters."""
    if vol.ndim == 5:
        t, z, c, y, x = vol.shape
        if timepoint is not None and (timepoint < 0 or timepoint >= t):
            raise ValueError(
                f"Timepoint {timepoint} out of bounds for volume with {t} timepoints"
            )
    elif vol.ndim == 4:
        z, c, y, x = vol.shape
        if timepoint is not None:
            raise ValueError("Timepoint should be None for 4D images")
    else:
        raise ValueError(
            f"Volume must be 5D (T,Z,C,Y,X) or 4D (Z,C,Y,X), got {vol.shape}"
        )

    if channel_id < 0 or channel_id >= c:
        raise ValueError(f"Channel {channel_id} out of bounds (have {c} channels)")

    if z_min is not None and (z_min < 0 or z_min >= z):
        raise ValueError(f"z_min {z_min} out of bounds (have {z} slices)")

    if z_max is not None and (z_max <= 0 or z_max > z):
        raise ValueError(f"z_max {z_max} out of bounds (have {z} slices)")

    if z_min is not None and z_max is not None and z_min >= z_max:
        raise ValueError(f"z_min ({z_min}) must be < z_max ({z_max})")

def verif_img_point(img,point):
    y, x = point.astype(int)

    y_max, x_max = img.shape
    if y < 0 or y >= y_max:
        raise ValueError(f"y coordinate {y} is out of bounds for stack height {y_max}.")
    if x < 0 or x >= x_max:
        raise ValueError(f"x coordinate {x} is out of bounds for stack width {x_max}.")

def verif_stack_point(stack, point):
    z, y, x = point.astype(int)

    z_max, y_max, x_max = stack.shape
    if z < 0 or z > z_max:
        raise ValueError(f"z coordinates of the point is out of bound for this stack")
    if y < 0 or y >= y_max:
        raise ValueError(f"y coordinate {y} is out of bounds for stack height {y_max}.")
    if x < 0 or x >= x_max:
        raise ValueError(f"x coordinate {x} is out of bounds for stack width {x_max}.")


def clamping_window_size(window_size, idx, img_size):
    idx_min = np.clip(idx - window_size//2, 0, img_size)
    idx_max = np.clip(idx + window_size//2, 0, img_size)

    if(idx_min == 0):
        idx_max = window_size
    if(idx_max == img_size):
        idx_min = img_size - window_size
    return idx_min, idx_max


def get_window_range(window_size, point, size):
    window_range = []
    for i in range(len(point)):
        coord = point[i].astype(int)        
        idx_min, idx_max = clamping_window_size(window_size, coord, size[i])
        window_range.append(slice(idx_min, idx_max))

    return tuple(window_range)

def get_windowed_coord(window_size,point,size):
    new_point = point.copy()
    for i in range(len(point)):
        coord = point[i].astype(int)        
        idx_min, idx_max = clamping_window_size(window_size, coord, size[i])
        new_point[i] -= idx_min

    return new_point

def get_pixel_size(vol_path):
    """
    Return the images pixel size in [px/um]
    """
    with tifffile.TiffFile(vol_path) as tif:
        imagej_metadata = tif.imagej_metadata
        z_pixel_size = imagej_metadata["spacing"]

        # Get first page
        page = tif.pages[0]
        y_res = page.tags["XResolution"].value
        y_pixel_size = y_res[1]/y_res[0]
        x_res = page.tags["XResolution"].value
        x_pixel_size = x_res[1]/x_res[0]

    return np.array([z_pixel_size,y_pixel_size,x_pixel_size])


def download_go_nuclear_model():
    model_name = "generic_plant_nuclei_3D"
    output_dir = Path("models") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "rdf.yaml": "https://zenodo.org/records/8432366/files/rdf.yaml",
        "weights_bioimageio.h5": "https://zenodo.org/records/8432366/files/stardist_weights.h5",
    }

    for fname, url in urls.items():
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(output_dir / fname, "wb") as f:
            f.write(r.content)

    with open(output_dir / "rdf.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg["config"]["stardist"]["config"], f, indent=2)

    with open(output_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(cfg["config"]["stardist"]["thresholds"], f, indent=2)

    print("Saved model to", output_dir)
from pathlib import Path
import os
import argparse
import pandas as pd
import sys
import pathlib
import traceback
import numpy as np
import tifffile
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization

def frame_composite_creation(frame,frame_bf,blobs_centers,annotation):
    max_radius = (blobs_centers[:, -1].max() /np.sqrt(3) + 1) * 1.2
    circle_mask = visualization.create_circle_mask(blobs_centers[:, :3], frame, radius = [max_radius])
   
    if annotation:
        mask = np.zeros(frame.shape, dtype=frame.dtype)
        annotation_mask = visualization.add_texts(mask, texts=np.arange(len(blobs_centers)).astype(str), coords=blobs_centers[:, :3])
        composite = np.stack([frame, circle_mask,annotation_mask,frame_bf], axis=1)

    else:
        composite = np.stack([frame, circle_mask,frame_bf], axis=1)
    return composite

def channel_id_from_path(path):
    name = Path(path).name  # keep only the filename
    match = re.search(r'c(\d+)_centers\.csv$', name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse channel id from '{name}'. Expected pattern like C1_coord.csv")
    return int(match.group(1))

def redindexing(centers_path,vol_path,annotation):
    print("Loading data")
    centers_df = pd.read_csv(centers_path)
    vol = tifffile.imread(vol_path)

    if "index" not in centers_df.columns:
        centers_df = centers_df.reset_index(drop=False)   # creates "index" col
    else:
        centers_df["index"] = np.arange(len(centers_df))

    channel_id = channel_id_from_path(centers_path)
    channel_idx = channel_id - 1

    print("Reindexing images")
    if vol.ndim == 4:
        vol_c = vol[:, channel_idx]
        vol_bf = vol[:, -1]
        centers = centers_df[["Z","Y","X","R"]].values.astype(int)
        composite = frame_composite_creation(vol_c,vol_bf,centers,annotation=True)
    elif vol.ndim == 5:
        t, z,c, y, x = vol.shape

        channels_nb = 4 if annotation else 3
        composite = np.empty((t, z, channels_nb, y, x), dtype=np.uint8)
        for ti in range(vol.shape[0]):
            vol_c = vol[ti, :, channel_idx]
            vol_bf = vol[ti, :, -1]
            centers_df_t = centers_df[centers_df["T"] == ti]
            centers = centers_df_t[["Z","Y","X","R"]].values.astype(int)
            composite[ti] = frame_composite_creation(vol_c,vol_bf,centers,annotation=True)
    else:
        raise ValueError(f"Image should be in 4D (CZYX) or 5D (TCZYX)")

    print("Saving results")
    centers_df.to_csv(centers_path, index=False)
    vol_output_path = centers_path.parent / f"c{channel_id}_detection_img.tif"
    tifffile.imwrite(vol_output_path,composite,imagej=True)

    print("Reindexing completed")

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Blob detection with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,

    )
    
    parser.add_argument("--centers_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--vol_path", required=True, type=pathlib.Path,
                    help="Input TIFF file")
    parser.add_argument("--annotation",  type=bool, default = True,
                    help="Input TIFF file")
    
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        redindexing(
            args.centers_path,
            args.vol_path,
            args.annotation
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
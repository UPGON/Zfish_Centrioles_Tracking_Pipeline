import sys
import os
import argparse
import pathlib
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import traceback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils
from src.pairing import points_pairing
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

def get_volumes(vol,channel1_id,channel2_id ):
    if vol.ndim == 4:
        vol1 = vol[:, channel1_id]
        vol2 = vol[:, channel2_id]
        vol_bf = vol[:,-1]

    elif vol.ndim == 5:
        vol1 = vol[:, :, channel1_id]
        vol2 = vol[:, :, channel2_id]
        vol_bf  = vol[:,:,-1]

    return vol1,vol2,vol_bf

def channel_id_from_path(path):
    name = Path(path).name  # keep only the filename
    match = re.search(r'c(\d+)_centers\.csv$', name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse channel id from '{name}'. Expected pattern like C1_coord.csv")
    return int(match.group(1))

def create_colocalizing_mask(frame, paired_centers_df):
    paired_centers1 = paired_centers_df[["Z", "Y", "X"]].values
    mask1 = visualization.create_circle_mask(paired_centers1, frame)
    return mask1


def create_annotated_pairing_detection(frame, centers_df, paired_centers_df):
    centers = centers_df[["Z", "Y", "X"]].values
    paired_centers = paired_centers_df[["Z", "Y", "X"]].values
    labels = centers_df["index"].astype(str).tolist()
    centers_mask = visualization.create_circle_mask(centers, frame)
    return visualization.add_texts(centers_mask, labels, paired_centers)



def save_colocalizing_mask(output_path, vol1, vol2, vol_bf, paired_centers1_df, paired_centers2_df):
    if vol1.ndim == 3:
        mask1 = create_colocalizing_mask(vol1, paired_centers1_df)
        mask2 = create_colocalizing_mask(vol2, paired_centers2_df)
        colocalizing_mask = np.stack([vol1, mask1, vol2, mask2,vol_bf], axis=1)
        axes = 'ZCYX'

    elif vol1.ndim == 4:
        t, z, y, x = vol1.shape
        colocalizing_mask = np.empty((t, z, 5, y, x), dtype=np.uint8)

        for ti in tqdm(range(t), desc="Creating colocalizing masks", unit="frame"):
            paired_t1 = paired_centers1_df[paired_centers1_df["T"] == ti]
            paired_t2 = paired_centers2_df[paired_centers2_df["T"] == ti]

            mask1 = create_colocalizing_mask(vol1[ti], paired_t1)
            mask2 = create_colocalizing_mask(vol2[ti], paired_t2)

            colocalizing_mask[ti] = np.stack([vol1[ti], mask1, vol2[ti], mask2,vol_bf[ti]], axis=1)
        axes = 'TZCYX'
    else:
        raise ValueError(f"Unexpected volume dimensions: {vol1.shape}")

    colocalizing_mask_path = output_path / "colocalizing_mask.tif"
    tifffile.imwrite(
        colocalizing_mask_path,
        colocalizing_mask,
        imagej=True,
        metadata={'axes': axes},
        compression='zlib'
    )


def save_annotated_pairing_detection(output_path, vol1, vol2,vol_bf, centers1_df, centers2_df,
                                      paired_centers1_df, paired_centers2_df):
    if vol1.ndim == 3:
        mask1 = create_annotated_pairing_detection(vol1, centers1_df, paired_centers1_df)
        mask2 = create_annotated_pairing_detection(vol2, centers2_df, paired_centers2_df)
        annotated_detection = np.stack([vol1, mask1, vol2, mask2,vol_bf], axis=1)
        axes = 'ZCYX'

    elif vol1.ndim == 4:
        t, z, y, x = vol1.shape
        annotated_detection = np.empty((t, z, 5, y, x), dtype=np.uint8)

        for ti in tqdm(range(t), desc="Creating annotated detections", unit="frame"):
            centers1_df_t         = centers1_df[centers1_df["T"] == ti]
            centers2_df_t         = centers2_df[centers2_df["T"] == ti]
            paired_centers1_df_t  = paired_centers1_df[paired_centers1_df["T"] == ti]
            paired_centers2_df_t  = paired_centers2_df[paired_centers2_df["T"] == ti]

            mask1 = create_annotated_pairing_detection(vol1[ti], centers1_df_t, paired_centers1_df_t)
            mask2 = create_annotated_pairing_detection(vol2[ti], centers2_df_t, paired_centers2_df_t)

            annotated_detection[ti] = np.stack([vol1[ti], mask1, vol2[ti], mask2,vol_bf[ti]], axis=1)
        axes = 'TZCYX'
    else:
        raise ValueError(f"Unexpected volume dimensions: {vol1.shape}")

    annotated_detection_path = output_path / "annotated_pairing_detection.tif"
    tifffile.imwrite(
        annotated_detection_path,
        annotated_detection,
        imagej=True,
        metadata={'axes': axes},
        compression='zlib'
    )


def save_figures(output_path, pairing_df, vol1, vol2,vol_bf,
                 centers1_df, centers2_df,
                 paired_centers1_df, paired_centers2_df):
    print("Saving figures...")
    output_path_img = output_path / "images"
    os.makedirs(output_path_img, exist_ok=True)

    save_colocalizing_mask(output_path_img, vol1, vol2,vol_bf,
                           paired_centers1_df, paired_centers2_df)
    save_annotated_pairing_detection(output_path_img, vol1, vol2,vol_bf,
                                     centers1_df, centers2_df,
                                     paired_centers1_df, paired_centers2_df)
    pairing_df.to_csv(output_path / "pairing_results.csv", index=False)


def redindexing(vol_path,pairing_path,centers1_path,centers2_path,channel1_id,channel2_id):
    print("Loading data")
    pairing_df = pd.read_csv(pairing_path)
    centers1_df = pd.read_csv(centers1_path)
    centers2_df = pd.read_csv(centers2_path)

    vol = tifffile.imread(vol_path)
    py_channel1_id = channel1_id - 1
    py_channel2_id = channel2_id - 1

    if "index" not in centers_df.columns:
        centers_df = centers_df.reset_index(drop=False)   # creates "index" col
    else:
        centers_df["index"] = np.arange(len(centers_df))

    vol1,vol2,vol_bf = get_volumes(vol, py_channel1_id, py_channel2_id)

    if len(pairing_df) > 0:
        paired_centers1_df = centers1_df.loc[pairing_df["idx1"].values].reset_index(drop=True)
        paired_centers2_df = centers2_df.loc[pairing_df["idx2"].values].reset_index(drop=True)

    else:
        print("No pairs found within distance threshold!")
        paired_centers1_df = pd.DataFrame(columns=centers1_df.columns)
        paired_centers2_df = pd.DataFrame(columns=centers2_df.columns)

    save_figures(output_path, pairing_df, vol1, vol2,vol_bf,
                 centers1_df, centers2_df,
                 paired_centers1_df, paired_centers2_df)
    
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
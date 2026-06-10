import sys
import os
import argparse
import pathlib
import time
import tifffile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import traceback
from skimage import morphology, filters
from skimage.filters import threshold_yen
from skimage.measure import label
from scipy import ndimage

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils
from pairing import pairing
from visualization import visualization
from visualization import plots

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

def get_intensity_at_centers(stack, centers_df):
    centers = centers_df[["Z", "Y", "X"]].values.astype(int)
    intensities = np.empty(len(centers), dtype=stack.dtype)

    for i, center in enumerate(centers):
        z, y, x = center
        intensities[i] = stack[z, y, x]

    return intensities

def get_diameter_at_centers(stack, centers_df,resolution):
    centers = centers_df[["Z", "Y", "X"]].values
    areas = np.empty(len(centers), dtype=np.float32)

    yen_stack = np.empty_like(stack)
    for z in np.unique(centers[:, 0]).astype(int):
        yen_stack[z] = stack[z] > threshold_yen(stack[z])

    for i, center in enumerate(centers):
        z, y, x = center.astype(int)
        yen_img = yen_stack[z]
        yen_labels, nshapes = label(yen_img, return_num=True)
        areas[i] = ndimage.sum(yen_img, labels=yen_labels, index=yen_labels[y, x])

        areas[i] = areas[i] * (resolution[-1] ** 2)  # Convert area from pixels to µm²
        areas[i] = np.sqrt(areas[i] / np.pi) *2

    return areas
    

def plot_areas_comparison(output_path,  paired_centers1_df,paired_centers2_df,unpaired_centers1_df,unpaired_centers2_df, channel1_name, channel2_name):
    if "area" not in paired_centers1_df.columns or "area" not in paired_centers1_df.columns:
        print("Area data not found in centers DataFrames: can't create area comparison plot")
        return

    paired_areas1 = paired_centers1_df["area"].values
    paired_areas2 = paired_centers2_df["area"].values

    unpaired_areas1 = unpaired_centers1_df["area"].values
    unpaired_areas2 = unpaired_centers2_df["area"].values

    plots.plot_comparison_box_plot(
        values = [paired_areas1, unpaired_areas1, paired_areas2, unpaired_areas2],
        labels = [channel1_name,channel1_name, channel2_name,channel2_name],
        linestyle = ["-", "--", "-", "--"],
        legends = {"paired": "-", "unpaired": "--"},
        title="Comparison of areas at detected foci",
        unit="Area [px]",
        colors =['forestgreen', 'darkseagreen','darkslateblue', 'mediumpurple'],
        output_path=output_path / "area_comparison_plot",
        show=False
    )

def plot_intensity_comparison(output_path,  paired_centers1_df,paired_centers2_df,unpaired_centers1_df,unpaired_centers2_df, channel1_name, channel2_name):
    if "intensity" not in paired_centers1_df.columns or "intensity" not in paired_centers1_df.columns:
        print("Intensity data not found in centers DataFrames: can't create intensity comparison plot")
        return

    paired_intensities1 = paired_centers1_df["intensity"].values
    paired_intensities2 = paired_centers2_df["intensity"].values

    unpaired_intensities1 = unpaired_centers1_df["intensity"].values
    unpaired_intensities2 = unpaired_centers2_df["intensity"].values

    plots.plot_comparison_box_plot(
        values = [paired_intensities1, unpaired_intensities1, paired_intensities2, unpaired_intensities2],
        labels = [channel1_name,channel1_name, channel2_name,channel2_name],
        linestyle = ["-", "--", "-", "--"],
        legends = {"paired": "-", "unpaired": "--"},
        title="Comparison of intensities at detected foci",
        unit="Intensity",
        colors =['forestgreen', 'darkseagreen','darkslateblue', 'mediumpurple'],
        output_path=output_path / "intensity_comparison_plot",
        show=False
    )

def plot_number_foci(output_path, centers1_df, centers2_df, channel1_name, channel2_name):
    counts = np.array([len(centers1_df.index), len(centers2_df.index)])
    plots.plot_count(
        values=counts,
        labels=[channel1_name, channel2_name],
        title="Number of detected foci",
        colors=['darkseagreen', 'mediumpurple'],
        output_path=output_path / "detection_count_plot",
        show=False
    )


def plot_detection_proportion(output_path, centers1_df, centers2_df, channel1_name, channel2_name):
    counts = np.array([len(centers1_df.index), len(centers2_df.index)])
    if counts.sum() == 0:
        print("No paired spots detected: can't create detection proportion plot")
        return
    proportions = counts / counts.sum()
    plots.plot_proportions(
        values=proportions,
        labels=[channel1_name, channel2_name],
        title="Proportion of detected foci",
        colors=['darkseagreen', 'mediumpurple'],
        output_path=output_path / "detection_proportion_plot",
        show=False
    )


def plot_colocalization_proportion(output_path, paired_centers1_df, centers1_df,
                                    paired_centers2_df, centers2_df,
                                    channel1_name, channel2_name):
    counts = np.array([len(paired_centers1_df.index), len(paired_centers2_df.index)])
    total  = np.array([len(centers1_df.index),len(centers2_df.index)])
    if 0 in total:
        print("No spots detected: can't create colocalization proportion plot")
        return
    plots.plot_proportions(
        values=counts / total,
        labels=[channel1_name, channel2_name],
        title="Proportion of colocalizing foci",
        colors=['darkseagreen', 'mediumpurple'],
        output_path=output_path / "colocalization_proportion_plot",
        show=False
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


def save_plots(output_path, unrestricted_pairing_df,
               paired_centers1_df, paired_centers2_df,
                unpaired_centers1_df,unpaired_centers2_df,
               centers1_df, centers2_df,
               channel1_name, channel2_name, max_pairing_distance):
    print("Saving plots...")
    output_path_plots = output_path / "plots"
    os.makedirs(output_path_plots, exist_ok=True)

    if len(unrestricted_pairing_df) > 0:
        plots.plot_zoomed_histogram(
            unrestricted_pairing_df["dist"],
            zooming_threshold=max_pairing_distance * 3,
            threshold=max_pairing_distance,
            title="Nearest neighbor distances",
            unit="Distance [µm]",
            output_path=output_path_plots / "pairing_distance_plot",
            show=False
        )

    plot_detection_proportion(output_path_plots, centers1_df, centers2_df,
                              channel1_name, channel2_name)
    plot_colocalization_proportion(output_path_plots,
                                   paired_centers1_df, centers1_df,
                                   paired_centers2_df, centers2_df,
                                   channel1_name, channel2_name)
    plot_number_foci(output_path_plots, centers1_df, centers2_df,
                              channel1_name, channel2_name)
    plot_areas_comparison(output_path_plots, paired_centers1_df,paired_centers2_df,unpaired_centers1_df,unpaired_centers2_df, channel1_name, channel2_name)
    plot_intensity_comparison(output_path_plots, paired_centers1_df,paired_centers2_df,unpaired_centers1_df,unpaired_centers2_df, channel1_name, channel2_name)


def save_statistics(output_path,centers1,centers2, pairing_df,paired_centers1_df,paired_centers2_df, max_pairing_distance):
    pd.DataFrame([{
        "pairs": len(pairing_df["dist"]),
        "ratio ct/cs": len(centers1.index)/len(centers2.index),
        "nb_of_paired_centers1": len(paired_centers1_df.index),
        "nb_of_paired_centers2": len(paired_centers2_df.index),
        "proportion_of_paired_centers1": (len(paired_centers1_df.index)/len(centers1.index)),
        "proportion_of_paired_centers2": (len(paired_centers2_df.index)/len(centers2.index)),
        "max_pairing_distance": max_pairing_distance,
        "mean_distance":   pairing_df["dist"].mean(),
        "median_distance": pairing_df["dist"].median(),
        "min_distance":    pairing_df["dist"].min(),
        "max_distance":    pairing_df["dist"].max(),
    }]).to_csv(output_path / "statistics.csv", index_label="index")


def save_results(output_path, vol1, vol2,vol_bf, pairing_df, unrestricted_pairing_df,
                 centers1_df, centers2_df, max_pairing_distance,
                 channel1_name, channel2_name):

    if len(pairing_df) > 0:
        paired_centers1_df = centers1_df.loc[pairing_df["idx1"].values].reset_index(drop=True)
        paired_centers2_df = centers2_df.loc[pairing_df["idx2"].values].reset_index(drop=True)
        unpaired_centers1_df = centers1_df.drop(pairing_df["idx1"].values).reset_index(drop=True)
        unpaired_centers2_df = centers2_df.drop(pairing_df["idx2"].values).reset_index(drop=True)

    else:
        print("No pairs found within distance threshold!")
        paired_centers1_df = pd.DataFrame(columns=centers1_df.columns)
        paired_centers2_df = pd.DataFrame(columns=centers2_df.columns)

    os.makedirs(output_path, exist_ok=True)

    save_figures(output_path, pairing_df, vol1, vol2,vol_bf,
                 centers1_df, centers2_df,
                 paired_centers1_df, paired_centers2_df)
    save_plots(output_path, unrestricted_pairing_df,
               paired_centers1_df, paired_centers2_df,
                unpaired_centers1_df,unpaired_centers2_df,
               centers1_df, centers2_df,
               channel1_name, channel2_name, max_pairing_distance)
    save_statistics(output_path, centers1_df,centers2_df, pairing_df,
                    paired_centers1_df,paired_centers2_df, 
                    max_pairing_distance)


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

def pairing_centers(vol, centers1_df, centers2_df, max_pairing_distance):
    z, c, y, x = vol.shape[-4:]

    if vol.ndim == 4:
        print("Processing single frame (4D volume)...")
        unrestricted_pairing_df = pairing.pairing_points_df(centers1_df, centers2_df, 1e10)
        pairing_df              = pairing.pairing_points_df(centers1_df, centers2_df, max_pairing_distance)

    elif vol.ndim == 5:
        print(f"Processing {vol.shape[0]} frames (5D volume)...")
        unrestricted_pairing_df = pairing.temporal_pairing_points_df(centers1_df, centers2_df, 1e10)
        pairing_df              = pairing.temporal_pairing_points_df(centers1_df, centers2_df, max_pairing_distance)
    else:
        raise ValueError(f"Volume must be 4D or 5D, got shape {vol.shape}")

    return  pairing_df, unrestricted_pairing_df


def colocalization_analysis(vol_path, centers1_path, centers2_path, output_path,
                             channel1_id, channel2_id, max_pairing_distance,
                             channel1_name, channel2_name):
    start_time = time.time()

    print(f"Loading volume from {vol_path}...")
    vol = tifffile.memmap(vol_path)
    py_channel1_id = channel1_id - 1
    py_channel2_id = channel2_id - 1

    utils.verify_input(vol, py_channel1_id)
    utils.verify_input(vol, py_channel2_id)

    print("Loading center coordinates...")
    centers1_df = pd.read_csv(centers1_path)
    centers2_df = pd.read_csv(centers2_path)

    if "index" not in centers1_df.columns:
        centers1_df = centers1_df.reset_index(drop=False)   # creates "index" col
    else:
        centers1_df = centers1_df.reset_index(drop=True)    # just resets df.index

    if "index" not in centers2_df.columns:
        centers2_df = centers2_df.reset_index(drop=False)
    else:
        centers2_df = centers2_df.reset_index(drop=True)

    resolution = utils.get_pixel_size(vol_path)
    centers1_df[["Zum", "Yum", "Xum"]] = centers1_df[["Z", "Y", "X"]] * resolution
    centers2_df[["Zum", "Yum", "Xum"]] = centers2_df[["Z", "Y", "X"]] * resolution

    pairing_df, unrestricted_pairing_df = pairing_centers(
        vol, centers1_df, centers2_df, max_pairing_distance
    )

    vol1,vol2,vol_bf = get_volumes(vol, py_channel1_id, py_channel2_id)

    centers1_df["intensity"] = get_intensity_at_centers(vol1, centers1_df)
    centers2_df["intensity"] = get_intensity_at_centers(vol2, centers2_df)
    centers1_df["area"] = get_diameter_at_centers(vol1, centers1_df,resolution)
    centers2_df["area"] = get_diameter_at_centers(vol2, centers2_df,resolution)

    save_results(output_path, vol1, vol2,vol_bf, pairing_df, unrestricted_pairing_df,
                 centers1_df, centers2_df, max_pairing_distance,
                 channel1_name, channel2_name)

    print(f"Colocalization analysis complete — took {time.time() - start_time:.2f}s")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Colocalization analysis for 4D and 5D microscopy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--vol_path",             required=True,  type=pathlib.Path)
    parser.add_argument("--centers1_path",         required=True,  type=pathlib.Path)
    parser.add_argument("--centers2_path",         required=True,  type=pathlib.Path)
    parser.add_argument("--output_path",           required=True,  type=pathlib.Path)
    parser.add_argument("--channel1_id",           required=True,  type=int)
    parser.add_argument("--channel2_id",           required=True,  type=int)
    parser.add_argument("--max_pairing_distance",  required=True,  type=float)
    parser.add_argument("--channel1_name",         type=str, default="Cetn2Eos")
    parser.add_argument("--channel2_name",         type=str, default="CenSpark")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        colocalization_analysis(
            args.vol_path, args.centers1_path, args.centers2_path,
            args.output_path, args.channel1_id, args.channel2_id,
            args.max_pairing_distance, args.channel1_name, args.channel2_name
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
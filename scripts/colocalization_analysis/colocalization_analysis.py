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
from skimage.feature import blob_dog, blob_log
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from pairing import pairing
from visualization import visualization
from detection.blob_detection import blob_detection


def save_colocalising_mask(output_path, vol1, vol2, paired_centers1_df, paired_centers2_df):
    z,y,x = vol1.shape

    mask1 = visualization.create_circle_mask(paired_centers1_df[["Z","Y","X"]].values, [z,y,x])
    mask2 = visualization.create_circle_mask(paired_centers2_df[["Z","Y","X"]].values,[z,y,x])

    pairing_centers_vol = np.stack([vol1, mask1, vol2,mask2], axis=1) 

    pairing_mask_path = output_path / "pairing_mask.tif"

    tifffile.imwrite(pairing_mask_path, 
                pairing_centers_vol,
                imagej=True,
                compression='zlib'
                )

def save_annotate_pairing_detection(output_path, pairing_df, vol1,vol2,centers1_df, centers2_df):
    centers1 = centers1_df[["Z","Y","X"]].values
    centers2 = centers2_df[["Z","Y","X"]].values

    centers_mask1 = visualization.create_circle_mask(centers1, vol1.shape)
    centers_mask2 = visualization.create_circle_mask(centers2, vol2.shape)

    list_pair_idx = pairing_df[["idx1","idx2"]].values
    for i in range(len(list_pair_idx)):
        idx1, idx2 = list_pair_idx[i].astype(int)
        visualization.add_text(centers_mask1,str(i), centers1[idx1])
        visualization.add_text(centers_mask2,str(i), centers2[idx2])

    annotated_vol = np.stack([vol1, centers_mask1, vol2,centers_mask2], axis=1) 

    annotated_pairing_detection_path = output_path / "annotated_pairing_detection.tif"

    tifffile.imwrite(annotated_pairing_detection_path, 
                    annotated_vol,
                    imagej=True,
                    compression='zlib'
                    )


def plot_detection_proportion(output_path, paired_centers1_df, paired_centers2_df, channel1_name,channel2_name):
    counts = np.array([len(paired_centers1_df.index), len(paired_centers2_df.index)])
    if(counts.sum() == 0):
        print("No spots detected: can't create plots")
        return
    
    proportions = counts/counts.sum()
    labels = [channel1_name, channel2_name]
    title = "Proprotion of detected spots"
    colors = ['darkseagreen', 'mediumpurple']
    output_path_plots_proprotion = output_path / "detection_proporstion_plot"
    visualization.plot_proportions(values = proportions,labels = labels, title = title, colors = colors,output_path=output_path_plots_proprotion,show=False)


def plot_colocalization_proportion(output_path, paired_centers1_df,centers1_df,paired_centers2_df,centers2_df,channel1_name,channel2_name ):
    counts = np.array([ len(paired_centers1_df.index), len(paired_centers2_df.index)])
    total = np.array([ len(centers1_df.index), len(centers2_df.index)])
    
    if(0 in total):
        print("No spots detected: can't create plots")
        return 
    
    proportion = counts/total
    labels = [channel1_name, channel2_name]
    title = "Proprotion of colocalizing spots"
    colors = ['darkseagreen', 'mediumpurple']
    output_path_plots_proprotion = output_path / "coloclization_proportion_plot"

    visualization.plot_proportions(values = proportion,labels = labels, title = title,colors =  colors,output_path=output_path_plots_proprotion,show=False)

    
def colocalization_analysis(
    vol_path,
    centers1_path,
    centers2_path,
    output_path,
    channel1_id,
    channel2_id,
    resolution,
    max_pairing_distance,
    channel1_name,
    channel2_name

):
    """Run the colocalisation analysis based on the results of the segmentation

        Args:
            resolution: A list containing the depth, height and width resolution
        

    """
    print("Loading the image")
    vol = tifffile.memmap(vol_path)

    centers1_df = pd.read_csv(centers1_path)
    centers2_df = pd.read_csv(centers2_path)

    print("Pairing the centers")
    unrestricted_pairing_df = pairing.pairing_points_df(centers1_df, centers2_df, 1e10)

    pairing_df = pairing.pairing_points_df(centers1_df, centers2_df, max_pairing_distance)

    paired_centers1_df = centers1_df.iloc[pairing_df["idx1"]]
    paired_centers2_df = centers2_df.iloc[pairing_df["idx2"]]

    os.makedirs(output_path, exist_ok=True)

    print("Creating visualisation images")
    output_path_img = output_path / "images"
    os.makedirs(output_path_img, exist_ok=True)

    save_colocalising_mask(output_path_img, vol[:,channel1_id], vol[:,channel2_id], paired_centers1_df,paired_centers2_df)
    save_annotate_pairing_detection(output_path_img, pairing_df, vol[:,channel1_id], vol[:,channel2_id], centers1_df, centers2_df)

    print("Creating plots")
    output_path_plots = output_path / "plots"
    os.makedirs(output_path_plots, exist_ok=True)

    visualization.plot_zoomed_histogram(unrestricted_pairing_df["dist"], zooming_threshold=max_pairing_distance*3, threshold=max_pairing_distance, title = "Nearest neigbour", unit = "Distance [um]",output_path=output_path_plots/"pairing_distance_plot",show=False)
    plot_detection_proportion(output_path_plots,paired_centers1_df,paired_centers2_df, channel1_name, channel2_name)
    plot_colocalization_proportion(output_path_plots, paired_centers1_df,centers1_df, paired_centers2_df,centers2_df, channel1_name, channel2_name)

    print("Process finished")

if __name__ == "__main__":
    ### TODO: add resolution so that I can transform coord in um
    # TODO change all the mention in mi by um 

    parser = argparse.ArgumentParser(
        description="Run the colocaliuzation analysis based on the given results"
    )
    parser.add_argument("--vol_path", required=True, type=pathlib.Path)
    parser.add_argument("--centers1_path", required=True, type=pathlib.Path)
    parser.add_argument("--centers2_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--channel1_id", required=True, type=int)
    parser.add_argument("--channel2_id", required=True, type=int)
    parser.add_argument(
        "--resolution",
        required=True,
        type=float,
        nargs=3,
        metavar=("DZ", "DY", "DX"),
        help="Voxel size in microns as three floats: depth, height, width",
    )
    parser.add_argument("--max_pairing_distance", required=True, type=float)
    parser.add_argument("--channel1_name", nargs="?", type=str, default = "Cetn2Eos", help="Name of channel 1 (default: Cetn2Eos)")
    parser.add_argument("--channel2_name", nargs="?", type=str, default = "CenSpark", help="Name of channel 2 (default: CenSpark)")
    args = parser.parse_args()

    try:
        print(f"test {args.vol_path}")
        colocalization_analysis(
            args.vol_path,
            args.centers1_path,
            args.centers2_path,
            args.output_path,
            args.channel1_id,
            args.channel2_id,
            args.resolution,
            args.max_pairing_distance,
            args.channel1_name,
            args.channel2_name
        )
    except Exception as e:
        print(f"Error detected: {e}", file=sys.stderr)
        sys.exit(1)

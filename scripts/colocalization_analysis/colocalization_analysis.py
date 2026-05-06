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

project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from utils import utils
from pairing import pairing
from visualization import visualization


def create_colocalizing_mask(frame1, frame2, paired_centers1_df, paired_centers2_df):
    """Create visualization mask for colocalized spots.
    
    Args:
        vol1: Channel 1 volume (Z, Y, X)
        vol2: Channel 2 volume (Z, Y, X)
        paired_centers1: Paired centers from channel 1 (N, 3) [Z, Y, X]
        paired_centers2: Paired centers from channel 2 (N, 3) [Z, Y, X]
    
    Returns:
        4-channel composite (Z, C=4, Y, X)
    """
    z, y, x = frame1.shape
    paired_centers1 = paired_centers1_df[["Z", "Y", "X"]].values
    paired_centers2 = paired_centers2_df[["Z", "Y", "X"]].values
    mask1 = visualization.create_circle_mask(paired_centers1, [z, y, x])
    mask2 = visualization.create_circle_mask(paired_centers2, [z, y, x])
    
    # Stack as (Z, C, Y, X) where C=4: [vol1, mask1, vol2, mask2]
    pairing_centers_vol = np.stack([frame1, mask1, frame2, mask2], axis=1)
    
    return pairing_centers_vol


def create_annotated_pairing_detection(frame1, frame2, centers1_df, centers2_df,paired_centers1_df, paired_centers2_df):
    """Create annotated visualization with pairing labels.
    
    Args:
        vol1: Channel 1 volume (Z, Y, X)
        vol2: Channel 2 volume (Z, Y, X)
        pairing_df: DataFrame with columns [idx1, idx2, dist]
        centers1: All centers from channel 1 (N, 3) [Z, Y, X]
        centers2: All centers from channel 2 (N, 3) [Z, Y, X]
    
    Returns:
        4-channel annotated composite (Z, C=4, Y, X)
    """
    centers1 = centers1_df[["Z", "Y", "X"]].values
    centers2 = centers2_df[["Z", "Y", "X"]].values
    centers_mask1 = visualization.create_circle_mask(centers1, frame1.shape)
    centers_mask2 = visualization.create_circle_mask(centers2, frame2.shape)
    
    # Add text labels for paired spots
    for i in range(len(paired_centers1_df)):
        centers_mask1 = visualization.add_text(centers_mask1, str(i), paired_centers1_df.iloc[i][["Z", "Y", "X"]].values)
        centers_mask2 = visualization.add_text(centers_mask2, str(i), paired_centers2_df.iloc[i][["Z", "Y", "X"]].values)
    
    # Stack as (Z, C, Y, X) where C=4: [vol1, mask1, vol2, mask2]
    annotated_vol = np.stack([frame1, centers_mask1, frame2, centers_mask2], axis=1)
    
    return annotated_vol


def save_colocalizing_mask(output_path, vol1, vol2, paired_centers1_df, paired_centers2_df):
    """Save colocalizing mask to file.
    
    Args:
        output_path: Output directory
        vol1: Channel 1 volume (Z, Y, X) or (T, Z, Y, X)
        vol2: Channel 2 volume (Z, Y, X) or (T, Z, Y, X)
        paired_centers1_df: DataFrame with paired centers from channel 1
        paired_centers2_df: DataFrame with paired centers from channel 2
    """
    if vol1.ndim == 3:
        # 3D case: (Z, Y, X)
        colocalizing_mask = create_colocalizing_mask(vol1, vol2, paired_centers1_df, paired_centers2_df)
        axes = 'ZCYX'
    elif vol1.ndim == 4:
        # 4D case: (T, Z, Y, X)
        t, z, y, x = vol1.shape
        colocalizing_mask = np.empty((t, z, 4, y, x), dtype=np.uint8)
        
        for ti in tqdm(range(t), desc="Creating colocalizing masks", unit="frame"):
            # Filter centers for this timepoint
            paired_t1 = paired_centers1_df[paired_centers1_df["T"] == ti]
            paired_t2 = paired_centers2_df[paired_centers2_df["T"] == ti]
            
            colocalizing_mask[ti] = create_colocalizing_mask(
                vol1[ti], vol2[ti], paired_t1, paired_t2
            )
        axes = 'TZCYX'
    else:
        raise ValueError(f"Unexpected volume dimensions: {vol1.shape}")
    
    # Save
    colocalizing_mask_path = output_path / "colocalizing_mask.tif"
    tifffile.imwrite(
        colocalizing_mask_path,
        colocalizing_mask,
        imagej=True,
        metadata={'axes': axes},
        compression='zlib'
    )


def save_annotated_pairing_detection(output_path, vol1, vol2, centers1_df, centers2_df,paired_centers1_df, paired_centers2_df):
    """Save annotated pairing detection to file.
    
    Args:
        output_path: Output directory
        pairing_df: DataFrame with pairing information
        vol1: Channel 1 volume (Z, Y, X) or (T, Z, Y, X)
        vol2: Channel 2 volume (Z, Y, X) or (T, Z, Y, X)
        centers1_df: DataFrame with all centers from channel 1
        centers2_df: DataFrame with all centers from channel 2
    """
    if vol1.ndim == 3:
        # 3D case: (Z, Y, X)
        annotated_detection = create_annotated_pairing_detection(
            vol1, vol2, centers1_df, centers2_df,paired_centers1_df, paired_centers2_df
        )
        axes = 'ZCYX'
    elif vol1.ndim == 4:
        # 4D case: (T, Z, Y, X)
        t, z, y, x = vol1.shape
        annotated_detection = np.empty((t, z, 4, y, x), dtype=np.uint8)
        
        for ti in tqdm(range(t), desc="Creating annotated detections", unit="frame"):
            # Filter for this timepoint
            centers1_df_t = centers1_df[centers1_df["T"] == ti]
            centers2_df_t = centers2_df[centers2_df["T"] == ti]

            paired_centers1_df_t = paired_centers1_df[paired_centers1_df["T"] == ti]
            paired_centers2_df_t = paired_centers2_df[paired_centers2_df["T"] == ti]
        
        
            annotated_detection[ti] = create_annotated_pairing_detection(
                vol1[ti], vol2[ti], centers1_df_t, centers2_df_t,paired_centers1_df_t, paired_centers2_df_t
            )
        axes = 'TZCYX'
    else:
        raise ValueError(f"Unexpected volume dimensions: {vol1.shape}")
    
    # Save
    annotated_detection_path = output_path / "annotated_pairing_detection.tif"
    tifffile.imwrite(
        annotated_detection_path,
        annotated_detection,
        imagej=True,
        metadata={'axes': axes},
        compression='zlib'
    )


def plot_detection_proportion(output_path, centers1_df, centers2_df, channel1_name, channel2_name):
    """Plot proportion of detected spots per channel."""
    counts = np.array([len(centers1_df.index), len(centers2_df.index)])
    
    if counts.sum() == 0:
        print("No paired spots detected: can't create detection proportion plot")
        return
    
    proportions = counts / counts.sum()
    labels = [channel1_name, channel2_name]
    title = "Proportion of paired spots"
    colors = ['darkseagreen', 'mediumpurple']
    output_path_plot = output_path / "detection_proportion_plot"
    
    visualization.plot_proportions(
        values=proportions,
        labels=labels,
        title=title,
        colors=colors,
        output_path=output_path_plot,
        show=False
    )


def plot_colocalization_proportion(
    output_path,
    paired_centers1_df,
    centers1_df,
    paired_centers2_df,
    centers2_df,
    channel1_name,
    channel2_name
):
    """Plot proportion of colocalized spots per channel."""
    counts = np.array([len(paired_centers1_df.index), len(paired_centers2_df.index)])
    total = np.array([len(centers1_df.index), len(centers2_df.index)])
    
    if 0 in total:
        print("No spots detected: can't create colocalization proportion plot")
        return
    
    proportion = counts / total
    labels = [channel1_name, channel2_name]
    title = "Proportion of colocalizing spots"
    colors = ['darkseagreen', 'mediumpurple']
    output_path_plot = output_path / "colocalization_proportion_plot"
    
    visualization.plot_proportions(
        values=proportion,
        labels=labels,
        title=title,
        colors=colors,
        output_path=output_path_plot,
        show=False
    )

def save_figures(output_path, pairing_df, vol1, vol2, centers1_df, centers2_df,paired_centers1_df, paired_centers2_df):
    print("Saving figures...")
    output_path_img = output_path / "images"
    os.makedirs(output_path_img, exist_ok=True)
    
    # Save masks
    save_colocalizing_mask(
        output_path_img, vol1, vol2, 
        paired_centers1_df, paired_centers2_df
    )
    
    save_annotated_pairing_detection(
        output_path_img, vol1, vol2,
        centers1_df, centers2_df,
        paired_centers1_df, paired_centers2_df
    )
    
    # Save pairing data
    pairing_csv_path = output_path / "pairing_results.csv"
    pairing_df.to_csv(pairing_csv_path, index=False)



def save_plots(output_path,unrestricted_pairing_df,paired_centers1_df,paired_centers2_df,centers1_df,centers2_df,channel1_name,channel2_name,max_pairing_distance):
    # Create plots
    print("Saving plots...")
    output_path_plots = output_path / "plots"
    os.makedirs(output_path_plots, exist_ok=True)
    
    # Distance histogram
    if len(unrestricted_pairing_df) > 0:
        visualization.plot_zoomed_histogram(
            unrestricted_pairing_df["dist"],
            zooming_threshold=max_pairing_distance * 3,
            threshold=max_pairing_distance,
            title="Nearest neighbor distances",
            unit="Distance [µm]",
            output_path=output_path_plots / "pairing_distance_plot",
            show=False
        )
    
    # Proportion plots
    plot_detection_proportion(
        output_path_plots,
        centers1_df,
        centers2_df,
        channel1_name,
        channel2_name
    )
    
    plot_colocalization_proportion(
        output_path_plots,
        paired_centers1_df,
        centers1_df,
        paired_centers2_df,
        centers2_df,
        channel1_name,
        channel2_name
    )
    
def save_statistics(output_path, pairing_df,max_pairing_distance):
    output_stats_path = output_path / f"statistics.csv"

    pd.DataFrame([{
        "max_pairing_distance":max_pairing_distance,
        "mean_distance": pairing_df["dist"].mean(),
        "median_distance": pairing_df["dist"].median(),
        "min_distance": pairing_df["dist"].min(),
        "max_distance": pairing_df["dist"].max(),
    }]).to_csv(output_stats_path, index_label="index")

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
    """Run colocalization analysis based on detection results.
    
    Args:
        vol_path: Path to volume TIFF (4D or 5D)
        centers1_path: Path to channel 1 centers CSV
        centers2_path: Path to channel 2 centers CSV
        output_path: Output directory
        channel1_id: Channel 1 ID
        channel2_id: Channel 2 ID
        resolution: Voxel size [dz, dy, dx] in microns
        max_pairing_distance: Maximum pairing distance in microns
        channel1_name: Name of channel 1
        channel2_name: Name of channel 2
    """
    start_time = time.time()
    
    print(f"Loading volume from {vol_path}...")
    vol = tifffile.memmap(vol_path)
    utils.verify_input(vol, channel1_id)
    utils.verify_input(vol, channel2_id)
    
    z, c, y, x = vol.shape[-4:]
    
    print("Loading center coordinates...")
    centers1_df = pd.read_csv(centers1_path)
    centers2_df = pd.read_csv(centers2_path)
    
    # Convert to microns
    resolution_mi_per_px = np.array(resolution) / np.array([z, y, x])
    centers1_df[["Zum", "Yum", "Xum"]] = (
        centers1_df[["Z", "Y", "X"]] * resolution_mi_per_px
    )
    centers2_df[["Zum", "Yum", "Xum"]] = (
        centers2_df[["Z", "Y", "X"]] * resolution_mi_per_px
    )
    
    # Extract channel volumes
    if vol.ndim == 4:
        # 4D: (Z, C, Y, X)
        print("Processing single frame (4D volume)...")
        vol1 = vol[:, channel1_id, :, :]
        vol2 = vol[:, channel2_id, :, :]
        
        # Compute pairing
        print(f"Pairing centers")
        unrestricted_pairing_df = pairing.pairing_points_df(
            centers1_df, centers2_df, 1e10
        )
        
        pairing_df = pairing.pairing_points_df(
            centers1_df, centers2_df, max_pairing_distance
        )
        
    elif vol.ndim == 5:
        # 5D: (T, Z, C, Y, X)
        print(f"Processing {vol.shape[0]} frames (5D volume)...")
        vol1 = vol[:, :, channel1_id, :, :]
        vol2 = vol[:, :, channel2_id, :, :]
        
        # Compute pairing
        print(f"Pairing centers")
        unrestricted_pairing_df = pairing.temporal_pairing_points_df(
            centers1_df, centers2_df, 1e10
        )
        
        pairing_df = pairing.temporal_pairing_points_df(
            centers1_df, centers2_df, max_pairing_distance
        )
    else:
        raise ValueError(f"Volume must be 4D or 5D, got shape {vol.shape}")
    
    # Get paired centers
    if len(pairing_df) > 0:
        paired_centers1_df = centers1_df.iloc[pairing_df["idx1"].values]
        paired_centers2_df = centers2_df.iloc[pairing_df["idx2"].values]
    else:
        print("No pairs found within distance threshold!")
        paired_centers1_df = pd.DataFrame(columns=centers1_df.columns)
        paired_centers2_df = pd.DataFrame(columns=centers2_df.columns)
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    #Save fiures
    save_figures(
        output_path,
        pairing_df,
        vol1,
        vol2,
        centers1_df,
        centers2_df,
        paired_centers1_df,
        paired_centers2_df
    )

    #Save plots
    save_plots(
        output_path,
        unrestricted_pairing_df,
        paired_centers1_df,
        paired_centers2_df,
        centers1_df,
        centers2_df,
        channel1_name,
        channel2_name,
        max_pairing_distance
    )

    #Save statistics
    save_statistics(
        output_path,
        pairing_df,
        max_pairing_distance
    )

    elapsed = time.time() - start_time
    
    print(f"Colocalization analysis complete")
    print(f"Took: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Colocalization analysis for 4D and 5D microscopy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 4D volume (Z, C, Y, X)
  python %(prog)s --vol_path data_4d.tif --centers1_path c1.csv --centers2_path c2.csv \\
    --output_path results/ --channel1_id 0 --channel2_id 2 \\
    --resolution 0.5 0.15 0.15 --max_pairing_distance 1.6
  
  # 5D volume (T, Z, C, Y, X)
  python %(prog)s --vol_path data_5d.tif --centers1_path c1.csv --centers2_path c2.csv \\
    --output_path results/ --channel1_id 0 --channel2_id 2 \\
    --resolution 0.5 0.15 0.15 --max_pairing_distance 1.6 \\
    --channel1_name "Cetn2" --channel2_name "CenSpark"
        """
    )
    
    parser.add_argument("--vol_path", required=True, type=pathlib.Path,
                       help="Path to volume TIFF file (4D or 5D)")
    parser.add_argument("--centers1_path", required=True, type=pathlib.Path,
                       help="Path to channel 1 centers CSV")
    parser.add_argument("--centers2_path", required=True, type=pathlib.Path,
                       help="Path to channel 2 centers CSV")
    parser.add_argument("--output_path", required=True, type=pathlib.Path,
                       help="Output directory")
    parser.add_argument("--channel1_id", required=True, type=int,
                       help="Channel 1 ID")
    parser.add_argument("--channel2_id", required=True, type=int,
                       help="Channel 2 ID")
    parser.add_argument(
        "--resolution",
        required=True,
        type=float,
        nargs=3,
        metavar=("DZ", "DY", "DX"),
        help="Voxel size in microns (depth, height, width)",
    )
    parser.add_argument("--max_pairing_distance", required=True, type=float,
                       help="Maximum pairing distance in microns")
    parser.add_argument("--channel1_name", type=str, default="Cetn2Eos",
                       help="Name of channel 1 (default: Cetn2Eos)")
    parser.add_argument("--channel2_name", type=str, default="CenSpark",
                       help="Name of channel 2 (default: CenSpark)")
    
    args = parser.parse_args()

    try:
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
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
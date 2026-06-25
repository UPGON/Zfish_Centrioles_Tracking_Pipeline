import sys
import pathlib
from pathlib import Path
import os
import argparse
import time
import tifffile
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from skimage.exposure import match_histograms
from skimage.transform import rescale
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization, plots
from utils import utils
import constants

COORDS_COLUMNS = ["Z","Y","X"]
COORDS_UM_COLUMNS = ["Zum","Yum","Xum"]

def pair_to_closest_center(centrioles, nucleus):
    tree = cKDTree(nucleus)
    distances, nucleus_idx = tree.query(centrioles,k=1)

    return  nucleus_idx, distances

def pair_to_closest_border(centrioles_df, nuclei_centers_df, nucleus_mask,scale):
    zz,yy,xx = nuclei_centers_df[COORDS_COLUMNS].T.values.astype(int)
    nuclei_centers_df["label_idx"] = nucleus_mask[zz,yy,xx]

    centrioles = centrioles_df[COORDS_COLUMNS].values.astype(int)
   
    binary_nucl = nucleus_mask > 0

    dist_outside, indices = distance_transform_edt(~binary_nucl, sampling=scale, return_indices=True)
    zz, yy, xx = centrioles[:, 0], centrioles[:, 1], centrioles[:, 2]
    distances = dist_outside[zz,yy,xx]

    # Get nearest nucleus index for each spot
    nearest_border_coords = indices[:,zz, yy, xx]
    
    nucleus_idx = map_border_to_nuclei_id(nearest_border_coords.T, nuclei_centers_df, nucleus_mask)

    return nucleus_idx, distances

def filter_pairing(nucleus_idx, distances,max_pairing_distances):
    initial_size = len(distances)
    centrioles_idx = np.arange(initial_size)
    if max_pairing_distances is not None:
        valid_mask = distances < max_pairing_distances
        distances = distances[valid_mask]
        nucleus_idx = nucleus_idx[valid_mask]
        centrioles_idx = centrioles_idx[valid_mask]
    
    return centrioles_idx, nucleus_idx, distances


def map_border_to_nuclei_id(nearest_border_coords,nuclei_df,nucleus_mask):
    nuclei_idx = np.empty(len(nearest_border_coords))
    for i in range(len(nearest_border_coords)):
        nearest_border_coord = nearest_border_coords[i]
        z,y,x = nearest_border_coord
        label_nucl_idx = nucleus_mask[z,y,x]

        candidate_nuclei = nuclei_df[nuclei_df["label_idx"] == label_nucl_idx].reset_index()
        diff_ex = np.abs(nearest_border_coord - candidate_nuclei[COORDS_COLUMNS].values)
        best_candidate_idx = diff_ex.mean(axis=1).argmin()
        nuclei_idx[i] = candidate_nuclei.iloc[best_candidate_idx]["index"]

    return nuclei_idx.astype(int)

def load_data(        
        vol_path,
        segm_path,
        centriole_center_path,
        nuclei_center_path
):
    vol = tifffile.imread(vol_path)
    segm_vol = tifffile.imread(segm_path)

    centrioles_df = pd.read_csv(centriole_center_path)

    nuclei_centers_df = pd.read_csv(nuclei_center_path)

    scale = utils.get_pixel_size(vol_path)

    centrioles_df[COORDS_UM_COLUMNS] = centrioles_df[COORDS_COLUMNS] * scale
    nuclei_centers_df[COORDS_UM_COLUMNS] = nuclei_centers_df[COORDS_COLUMNS] * scale

    return vol, segm_vol, centrioles_df, nuclei_centers_df

def save_pairing_img(output_path,vol, centrioles_centers, nuclei_centers,centrioles_idx, nuclei_idx):
    output_path_img = output_path / "images"
    os.makedirs(output_path_img, exist_ok=True)
    centrioles_centers_mask = visualization.create_circle_mask(centrioles_centers, vol[:,0])
    centrioles_labels = centrioles_idx.astype(str).tolist()
    centrioles_labels_coords = centrioles_centers[centrioles_idx]
    centrioles_centers_mask = visualization.add_texts(centrioles_centers_mask, texts = centrioles_labels, coords = centrioles_labels_coords)

    nuclei_centers_mask = visualization.create_circle_mask(nuclei_centers, vol[:,0])
    nuclei_labels = np.arange(len(nuclei_centers)).astype(str).tolist()
    nuclei_labels_coords = nuclei_centers[nuclei_idx]
    nuclei_centers_mask = visualization.add_texts(nuclei_centers_mask, texts = nuclei_labels, coords = nuclei_labels_coords)

    centrioles_lines_mask = np.zeros(vol[:,0].shape,dtype=vol[:,0].dtype)
    nuclei_lines_mask = np.zeros(vol[:,0].shape,dtype=vol[:,0].dtype)
    for i in range(len(centrioles_idx)):
        cz,cy,cx = centrioles_centers[centrioles_idx[i]].astype(int)
        nz,ny,nx = nuclei_centers[nuclei_idx[i]].astype(int)
        centrioles_lines_mask[cz] = cv2.line(centrioles_lines_mask[cz], (cx,cy),(nx,ny),color = 255)
        nuclei_lines_mask[nz] = cv2.line(nuclei_lines_mask[nz], (cx,cy),(nx,ny),color = 255)

    composite = np.stack([vol[:,0],vol[:,1],vol[:,2],vol[:,3], centrioles_centers_mask,centrioles_lines_mask,nuclei_centers_mask,nuclei_lines_mask], axis =1)

    output_annotated_mask = output_path_img/ "border_annotated_pairing_detection.tif"
    
    tifffile.imwrite(output_annotated_mask,
                     composite,
                     imagej = True,
                     metadata={"axes":"ZCYX"},
                     compression="zlib"
                     )
    
def save_foci_per_paired_nuclei(cent_per_nucl,output_path_plots):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cent_per_nucl)
    ax.set_title("Number of centrioles per paired nuclei")
    ax.set_xlabel("Number of centrioles per paired nuclei")
    ax.set_ylabel("Frequency")
    ax.legend([f"Mean {cent_per_nucl.mean():.1f}"], loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path_plots / "centrioles_per_paired_nuclei_plot.png", dpi=300)
    plt.close(fig)

def save_foci_per_tot_nuclei(cent_per_nucl_tot,output_path_plots):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cent_per_nucl_tot)
    ax.set_title("Number of centrioles per nuclei")
    ax.set_xlabel("Number of centrioles per nuclei")
    ax.set_ylabel("Frequency")
    ax.legend([f"Mean {cent_per_nucl_tot.mean():.1f}"], loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path_plots / "centrioles_per_tot_nuclei_plot.png", dpi=300)
    plt.close(fig)

def save_distance_histogram(distances,output_path_plots):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(distances,bins=100)
    ax.set_title("Foci distance to nearest nuclei border")
    ax.set_xlabel("Distance [um]")
    ax.set_ylabel("Frequency")
    ax.legend([f"Mean {distances.mean():.1f}"], loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path_plots / "distance_plot.png", dpi=300)
    plt.close(fig)

   
def save_plots(pairing_df, nuclei_centers_df,distances,max_pairing_distance, output_path):
    print("Saving plots")
    output_path_plots = output_path / "plots"
    os.makedirs(output_path_plots, exist_ok=True)

    cent_per_nucl, cent_per_tot_nucl = get_centriole_per_nuclei(pairing_df, nuclei_centers_df)

    save_foci_per_paired_nuclei(cent_per_nucl,output_path_plots)
    
    save_foci_per_tot_nuclei(cent_per_tot_nucl, output_path_plots)

    save_distance_histogram(distances, output_path_plots)

    if max_pairing_distance is not None:
        plots.plot_zoomed_histogram(
            values = distances,
            zooming_threshold= max_pairing_distance * 1.2,
            threshold= max_pairing_distance,
            title= "Foci distance to nearest nuclei border",
            unit = "Distance [um]",
            output_path=output_path_plots / "zoomed_distance_plot",
            show=False
        )

def  save_statisitics(pairing_df,centrioles_center_df,nuclei_centers_df, distances,max_pairing_distance,output_path):
    pairing_df.to_csv(output_path / "pairing_results.csv", index=False)

    cent_per_nucl, cent_per_tot_nucl = get_centriole_per_nuclei(pairing_df, nuclei_centers_df)
    paired_centriole_percentage, paired_nuclei_percentage = get_pairing_percentage(centrioles_center_df, nuclei_centers_df, pairing_df)

    pd.DataFrame([{
        "pairs": len(pairing_df.index),
        "mean distance": pairing_df["distances"].mean(),
        "mean unrestricted distance": np.mean(distances),
        "max_pairing_distance":max_pairing_distance,
        "mean nb of centriole per paired nuclei":cent_per_nucl.mean(),
        "mean nb of centriole per total nuclei":cent_per_tot_nucl.mean(),
        "Percentage of paired centrioles": paired_centriole_percentage,
        "Percentage of paired nuclei": paired_nuclei_percentage
    }]).to_csv(output_path / "statistics.csv", index_label="index")

def get_centriole_per_nuclei(pairing_df, nuclei_df):
    unpaired_nucl = nuclei_df.drop(pairing_df["nucl_idx"].values.astype(int))
    unpaired_ratio = np.zeros(len(unpaired_nucl.index))

    cent_per_nucl = pairing_df.groupby("nucl_idx").count()["c_idx"].values
    cent_per_nucl_tot = np.concatenate([cent_per_nucl,unpaired_ratio])

    return cent_per_nucl, cent_per_nucl_tot

def get_pairing_percentage(centrioles_df, nuclei_df, pairing_df):
    paired_centriole_percentage = len(pairing_df["c_idx"])/len(centrioles_df.index)
    paired_nuclei_percentage = len(np.unique(pairing_df["nucl_idx"]))/len(nuclei_df.index)

    return paired_centriole_percentage, paired_nuclei_percentage

def pairing_cent_nucl(
        vol_path,
        segm_path,
        centriole_center_path,
        nuclei_center_path,
        max_pairing_distance=None,
        output_path=None
):
    print("Data loading")
    vol, segm_vol, centrioles_df, nuclei_centers_df = load_data(
        vol_path,
        segm_path,
        centriole_center_path,
        nuclei_center_path
    )
    scale = utils.get_pixel_size(vol_path)

    print("Pair centrioles to closest nuclei")
    #nuclei_idx, distances = pair_to_closest_center(pairing_df[COORDS_UM_COLUMNS],nuclei_centers_df[COORDS_UM_COLUMNS])

    nuclei_idx, distances = pair_to_closest_border(centrioles_df, nuclei_centers_df,segm_vol[:,1],scale)

    centrioles_idx, nuclei_idx, filter_distance = filter_pairing(nuclei_idx,distances, max_pairing_distance) 

    ct_pair_data =np.stack([centrioles_idx, nuclei_idx,filter_distance], axis=1)
    pairing_cent_nucl = pd.DataFrame(ct_pair_data, columns=["c_idx","nucl_idx","distances"])


    if output_path is None:
            output_path = Path(f"pairing_res")
    os.makedirs(output_path, exist_ok=True)

    print("Save results")
    save_pairing_img(
        output_path,
        vol,
        centrioles_df[COORDS_COLUMNS].values,
        nuclei_centers_df[COORDS_COLUMNS].values,
        centrioles_idx,
        nuclei_idx
    )
    save_plots(pairing_cent_nucl, nuclei_centers_df,distances,max_pairing_distance,output_path)
    save_statisitics(pairing_cent_nucl,centrioles_df,nuclei_centers_df,distances,max_pairing_distance,output_path)
    print("Pairing finished")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pair centrioles with nuclei",
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
    
    parser.add_argument("--vol_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--segm_path", required=True,  type=pathlib.Path,
                       help="Channel to process")
    parser.add_argument("--centriole_center_path",  required=True, type=pathlib.Path,
                       help="Output directory")
    parser.add_argument("--nuclei_center_path", required=True, type=pathlib.Path,
                       help="Maximum radius [um] (default: 1.3)")
    parser.add_argument("--max_pairing_dist", type=float,
                       help="Maximum radius [um] (default: 1.3)")
    parser.add_argument("--output_path", type=pathlib.Path,
                       help="Maximum area in µm² for filtering")
    
    
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        pairing_cent_nucl(
            args.vol_path,
            args.segm_path,
            args.centriole_center_path,
            args.nuclei_center_path,
            args.max_pairing_dist,
            args.output_path
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
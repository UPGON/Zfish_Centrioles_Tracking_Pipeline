import btrack
import sys
import pathlib
from pathlib import Path
import argparse
import tifffile
import pandas as pd
import traceback
import os 
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import visualization
from utils import utils
from src.pairing import points_pairing

def track(objects, config_path, axis_lim, scale, max_search_radius = 7):
    with btrack.BayesianTracker() as tracker:

        # configure the tracker using a config file
        tracker.configure(config_path)

        # append the objects to be tracked
        tracker.append(objects)

        # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
        axis_max = axis_lim * scale
        tracker.volume = ((0, axis_max[2]), (0, axis_max[1]), (0, axis_max[0]))

        tracker.max_search_radius = max_search_radius

        # track them (in interactive mode)
        tracker.track(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks as a python list
        tracks = tracker.tracks

        # optional: get the data in a format for napari
        data, properties, graph = tracker.to_napari()

    return data, properties, graph

def get_objects_from_coords(coords):
    objects = []

    for idx, row in coords.iterrows():
        obj = btrack.btypes.PyTrackObject()
        obj.t = int(row['T'])
        obj.x = float(row['Xum'])
        obj.y = float(row['Yum'])
        obj.z = float(row['Zum'])
        obj.ID = idx  # Original index for reference
        
        objects.append(obj)

    return objects

def get_objects_from_seg(labels, img, scale):
    objects =  btrack.io.segmentation_to_objects(segmentation=labels, intensity_image=img, scale=scale)
    return objects

def create_composite(data,vol,channel_id):
    vol_c = vol[:,:,channel_id]
    vol_bf = vol[:,:,-1]
    track_id_vol = visualization.add_texts_4D(vol_c, data.astype(int), text_col = "track_id")
    taj_vol = visualization.draw_trajectories(vol_c,data)
    return np.stack([vol_c,taj_vol,track_id_vol,vol_bf],axis=2)

def save_results(output_path, data_df,composite):
    print("Saving results...")
    if output_path is None:
        output_path = Path("btrack_res")
    os.makedirs(output_path, exist_ok=True)

    output_data_path = output_path /"data.csv"
    data_df.to_csv(output_data_path)

    output_img_path = output_path /"track_vol.tif"
    tifffile.imwrite(output_img_path,composite,imagej=True,compression="zlib")


def btracking(input_path, vol_path,output_path,channel_id,config_path, max_search_radius):
    print("Loading data")
    scale = utils.get_pixel_size(vol_path)
    vol = tifffile.imread(vol_path)
    if input_path.suffix.lower() == ".tif":
        print("Segmented image was received")
        segm_vol = tifffile.imread(input_path)
        objects = get_objects_from_seg(segm_vol[:,:,1],segm_vol[:,:,0], scale)
    elif input_path.suffix.lower() == ".csv":
        print("Centers coordinates was received")
        coords = pd.read_csv(input_path)
        coords[["Zum", "Yum", "Xum"]] = coords[["Z", "Y", "X"]] * scale
        objects = get_objects_from_coords(coords)

    print("Tracking")
    _,z,_,y,x = vol.shape
    data, properties, graph  = track(objects, config_path,axis_lim= np.array([z,y,x]), scale = scale , max_search_radius = max_search_radius)
    data_df = pd.DataFrame(data,columns=["track_id","T","Zum","Yum","Xum"])
    data_df[["Z","Y","X"]] = (data_df[["Zum","Yum","Xum"]] /scale).astype(int)

    print("Creating composite")
    composite = create_composite(data_df,vol,channel_id)

    print("Saving")
    save_results(output_path,data_df,composite)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Track object using the Bayesian tracker Btrack",
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
    
    parser.add_argument("--input_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--vol_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--output_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--channel_id", required=True, type=int,
                       help="Channel to process")
    parser.add_argument("--config_path", required=True, type=pathlib.Path,
                       help="Path of the btrack config")
    parser.add_argument("--max_search_radius", type=float, default=7,
                       help="Max distance that an objects can move between frames")
    
    
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        btracking(
            args.input_path,
            args.vol_path,
            args.output_path,
            args.channel_id,
            args.config_path,
            args.max_search_radius,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
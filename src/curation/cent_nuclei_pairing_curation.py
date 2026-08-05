import sys
import os
import re
import argparse
import pathlib
import pandas as pd
from pathlib import Path
import traceback
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import tifffile
import math

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils
import constants
from pairing import cent_nucl_pairing

INSTRUCTIONS_FORMAT = ".txt"

def read_txt_path(txt_path):
    with open(txt_path) as file:
        lines = [line.rstrip() for line in file]
    return lines

# From this stack overflow post: https://stackoverflow.com/questions/16464279/how-can-i-best-assert-a-value-can-be-converted-to-int-in-python
def is_integery(val):
    if isinstance(val, (int)):  # actually integer values
        return True
    elif isinstance(val, float):  # some floats can be converted without loss
        return int(val) == float(val)
    elif not isinstance(val, str):  # we can't convert non-string
        return False
    else:    
        try:  # try/except is better then isdigit, because "-1".isdigit() - False
            int(val)
        except ValueError:
            return False  # someting non-convertible

        return True
    
def parse_reindexing_line(line):
    raw = line.strip()
    if not raw:
        return None

    tokens = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
    if len(tokens) != 2:
        raise ValueError(f"Pairing line must contain 2 coordinate values: '{line}'")
    return int(tokens[0]), int(tokens[1])

def repairing(instructions,pairing_df,centrioles_df, nuclei_df, segm, scale):
    binary_segm = segm[:,1] > 0
    distance_map = distance_transform_edt(binary_segm, sampling=scale)

    curated_pairing_df = pairing_df.copy()
    for line in instructions:
        if not line.strip():
            continue
        c_idx, nucl_idx = parse_reindexing_line(line)
        # Re-map the pairing index
        curated_pairing_df.loc[curated_pairing_df["c_idx"] == c_idx, "nucl_idx"] = nucl_idx

        # Get the distance between the centriole and the nucleus border
        nz, ny, nx = nuclei_df.iloc[nucl_idx][["Z", "Y", "X"]].astype(int)
        distance = math.dist(centrioles_df.iloc[c_idx][["Z", "Y", "X"]], nuclei_df.iloc[nucl_idx][["Z", "Y", "X"]])
        nuclei_distance_to_border = distance_map[nz,ny,nx]

        centriole_distance_to_border = distance - nuclei_distance_to_border
        curated_pairing_df.loc[curated_pairing_df["c_idx"] == c_idx,"distance_to_border"] = centriole_distance_to_border

    return curated_pairing_df

def curation(vol_path,segm_path,pairing_path, centriole_center_path, nuclei_center_path,instructions_path, output_path=None):
    pairing_df = pd.read_csv(pairing_path)
    centrioles_df = pd.read_csv(centriole_center_path)
    nuclei_df = pd.read_csv(nuclei_center_path)

    vol = tifffile.imread(vol_path)
    segm = tifffile.imread(segm_path)

    scale = utils.get_pixel_size(vol_path)

    instructions_files = [p for p in instructions_path.iterdir() if p.suffix.lower() == INSTRUCTIONS_FORMAT]

    for instruction_file in instructions_files:
        if instruction_file.stem.lower().startswith("pairing"):
            print(f"Processing {instruction_file.name} for re-pairing")
            instructions = read_txt_path(instruction_file)
            curated_pairing_df = repairing(instructions, pairing_df, centrioles_df, nuclei_df, segm, scale)

        else:
            raise ValueError(f"Unknown instruction file type: {instruction_file.name}")

    if output_path is None:
        output_path = Path(f"pairing_res")
    os.makedirs(output_path, exist_ok=True)

    print("Save results")
    cent_nucl_pairing.save_pairing_img(
        output_path,
        vol,
        centrioles_df[constants.COORDS_COLUMNS].values,
        nuclei_df[constants.COORDS_COLUMNS].values,
        curated_pairing_df["c_idx"].values.astype(int),
        curated_pairing_df["nucl_idx"].values.astype(int),
    )
    cent_nucl_pairing.save_plots(curated_pairing_df, nuclei_df,curated_pairing_df["distances"].values,output_path)
    cent_nucl_pairing.save_statisitics(curated_pairing_df,centrioles_df,nuclei_df,curated_pairing_df["distances"].values,output_path)
    print("Curation finished")

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="curate centriole-nuclei pairing data based on instructions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--vol_path", required=True, type=pathlib.Path,
                    help="Input TIFF file")
    parser.add_argument("--segm_path", required=True,  type=pathlib.Path,
                           help="Segmentation TIFF file")
    parser.add_argument("--pairing_path", required=True, type=pathlib.Path,
                    help="Input TIFF file")
    parser.add_argument("--centriole_center_path",  required=True, type=pathlib.Path,
                        help="Centriole centers CSV file")
    parser.add_argument("--nuclei_center_path", required=True, type=pathlib.Path,
                           help="Nuclei centers CSV file")
    parser.add_argument("--instructions_path", required=True, type=pathlib.Path,
                help="Input TIFF file")
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        curation(
            args.vol_path,
            args.segm_path,
            args.pairing_path,
            args.centriole_center_path,
            args.nuclei_center_path,
            args.instructions_path,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
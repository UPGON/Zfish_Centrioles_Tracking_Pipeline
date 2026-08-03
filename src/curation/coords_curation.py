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
import tifffile


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils
import constants

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

def parse_addition_line(line):
    raw = line.strip()
    if not raw:
        return None

    bracket_match = re.search(r"\[(.*)\]", raw)
    if not bracket_match:
        raise ValueError(f"Addition line must contain coordinates in brackets: '{line}'")

    inside = bracket_match.group(1)
    tokens = [p.strip() for p in re.split(r"[,\s]+", inside) if p.strip()]
    if len(tokens) < 3:
        raise ValueError(f"Addition line must contain at least 3 coordinate values: '{line}'")

    coords = []
    i = 0
    while i < len(tokens) and len(coords) < 3:
        token = tokens[i]
        if re.match(r"^[xyzXYZ]$", token):
            i += 1
            if i >= len(tokens):
                raise ValueError(f"Missing coordinate value after label '{token}' in line '{line}'")
            token = tokens[i]
        elif re.match(r"^[xyzXYZ][:=].+", token):
            token = re.sub(r"^[xyzXYZ][:=]", "", token)
        elif re.match(r"^[xyzXYZ].+", token) and not is_integery(token):
            token = re.sub(r"^[xyzXYZ]+", "", token)

        if not is_integery(token):
            raise ValueError(f"Coordinate value is not an integer: '{token}' in line '{line}'")
        coords.append(int(token))
        i += 1

    if len(coords) != 3:
        raise ValueError(f"Could not parse 3 coordinates from line: '{line}'")

    return coords


def removal(centers_df, instructions):
    assert all(is_integery(line) for line in instructions), "All lines in removal instruction files should be integers corresponding to the index of the spot to remove"
    idx_to_remove = [int(line) for line in instructions]
    return centers_df.drop(idx_to_remove)

def measure_features(centers,frame, window_size = 40):
    areas = np.empty(len(centers))
    intensities = np.empty(len(centers))

    for i,center in enumerate(centers):
        z,y,x = center.astype(int)
        window_range = utils.get_window_range(window_size, center[1:3], frame.shape[1:3])
        wz,wy,wx = utils.get_windowed_coord(window_size, center,frame.shape).astype(int)

        img = frame[z][window_range]
        gaussian_img = ndimage.gaussian_filter(img,sigma=1)
        yen_img = gaussian_img > threshold_otsu(gaussian_img)

        yen_labels = label(yen_img)
        areas[i] = ndimage.sum(yen_img, labels=yen_labels, index=yen_labels[wy,wx])
        intensities[i] = gaussian_img[wy,wx]

    return areas, intensities

def addition(centers_df, instructions, vol, window_size):
    coords = []
    for line in instructions:
        if not line.strip():
            continue
        coord = parse_addition_line(line)
        if coord is None:
            continue
        coords.append(coord)

    if coords:
        coords = np.array(coords)
        areas, intensities = measure_features(coords, vol, window_size)
        output_data = np.column_stack([np.arange(len(areas)),coords[:,0],coords[:,1],coords[:,2],areas,intensities])
        addition_df = pd.DataFrame(output_data, columns=["index","Z", "Y", "X", "Area", "Intensity"])
        centers_df = pd.concat([centers_df, addition_df])
    return centers_df

def curation(centers_path, instructions_path,vol_path,channel_id):
    centers_df = pd.read_csv(centers_path)

    instructions_files = [p for p in instructions_path.iterdir() if p.suffix.lower() == INSTRUCTIONS_FORMAT]

    for instruction_file in instructions_files:
        if instruction_file.stem.lower().startswith("remove"):
            print(f"Processing {instruction_file.name} for removal")
            instructions = read_txt_path(instruction_file)
            centers_df = removal(centers_df, instructions)

        elif instruction_file.stem.lower().startswith("addition") or instruction_file.stem.lower().startswith("add"):
            print(f"Processing {instruction_file.name} for addition")
            instructions = read_txt_path(instruction_file)
            vol = tifffile.imread(vol_path)
            window_size_px = int(constants.ROI_WINDOW_SIZE / utils.get_pixel_size(vol_path)[1])
            centers_df = addition(centers_df, instructions,vol[:,(channel_id-1)],window_size_px)


    centers_df["index"] = np.arange(len(centers_df))
    output_path = Path(f"{centers_path.stem}_curated{centers_path.suffix}")
    centers_df.to_csv(output_path, index=False)

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Blob detection with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,

    )
    
    parser.add_argument("--centers_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--instructions_path", required=True, type=pathlib.Path,
                    help="Input TIFF file")
    parser.add_argument("--vol_path", required=True, type=pathlib.Path,
                    help="Input TIFF file")
    parser.add_argument("--channel_id", required=True, type=int,
                    help="Input TIFF file")
    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        curation(
            args.centers_path,
            args.instructions_path,
            args.vol_path,
            args.channel_id
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
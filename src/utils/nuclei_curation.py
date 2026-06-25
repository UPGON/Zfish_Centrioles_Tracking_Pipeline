import sys
import os
import re
import argparse
import pathlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import traceback
import numpy as np
from skimage.filters import threshold_yen
from skimage.measure import label
from scipy import ndimage
import tifffile
from skimage.measure import label


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


INSTRUCTIONS_FORMAT = ".txt"
SEGM_CHANNEL_ID = 1

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


def removal(centers_df,segm_vol, instructions):
    assert all(is_integery(line) for line in instructions), "All lines in removal instruction files should be integers corresponding to the index of the spot to remove"
    idx_to_remove = [int(line) for line in instructions]
    
    binary_vol = segm_vol > 0
    label_vol = segm_vol.copy()#label(binary_vol)
    for remove_i in idx_to_remove:
        z,y,x = centers_df[["Z","Y","X"]].iloc[remove_i].astype(int)
        label_idx = label_vol[z,y,x]
        label_vol[label_vol == label_idx] = 0


    mask = ~centers_df.index.isin(idx_to_remove)
    centers_df_filtered = centers_df[mask].reset_index(drop=True)
    
    return centers_df_filtered, label_vol

def nuclei_curation(centers_path, segm_path, instructions_path):
    centers_df = pd.read_csv(centers_path)
    centers_df = centers_df.reset_index(drop=True)  # Ensure index is [0, 1, 2, ...]
    segm_vol = tifffile.imread(segm_path)

    instructions_files = [p for p in instructions_path.iterdir() if p.suffix.lower() == INSTRUCTIONS_FORMAT]

    for instruction_file in instructions_files:
        if instruction_file.stem.lower().startswith("remove"):
            print(f"Processing {instruction_file.name} for removal")
            instructions = read_txt_path(instruction_file)
            centers_df, segm_vol[:,SEGM_CHANNEL_ID] = removal(centers_df, segm_vol[:,SEGM_CHANNEL_ID], instructions)
            centers_df = centers_df.reset_index(drop=True)  # Reset index after removal
        else:
            raise ValueError("Instruction file should be named remove.txt")

    centers_df["index"] = np.arange(len(centers_df))
    output_data_path = Path(f"{centers_path.stem}_curated{centers_path.suffix}")
    centers_df.to_csv(output_data_path, index=False)
    
    output_vol_path = Path(f"{segm_path.stem}_curated{segm_path.suffix}")
    tifffile.imwrite(
        output_vol_path,
        segm_vol,
        imagej=True,
        metadata={"axes":"ZCYX"}
    )

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Blob detection with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,

    )
    
    parser.add_argument("--centers_path", required=True, type=pathlib.Path,
                       help="Input TIFF file")
    parser.add_argument("--segm_path", required=True, type=pathlib.Path,
                help="Input TIFF file")
    parser.add_argument("--instructions_path", required=True, type=pathlib.Path,
                    help="Input TIFF file")

    return parser

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    try:
        nuclei_curation(
            args.centers_path,
            args.segm_path,
            args.instructions_path,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
import sys
import os 
import argparse
import pathlib
import time
import math
import tifffile
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log
from tqdm import tqdm

def center_mask(blobs, shape):
    mask = np.zeros(shape,dtype=np.uint8)
    for p in blobs:
        #[z, y, x, r] = [int(p[0]), int(p[1]), int(p[2]), int(p[3] * math.sqrt(2))]
        [z, y, x, r] = [int(p[0]), int(p[1]), int(p[2]), 4]
        cv2.circle(mask[z], center=(x,y), radius= r , color=200, thickness=1)
    return mask

def apply_algo(img,algorithm, max_sigma, threshold):
    match algorithm:
        case "log":
            print("Log algorithm processing..")
            return blob_log(img, max_sigma=max_sigma, threshold=threshold)
        case "dog": 
            print("Dog algorithm processing..")
            return blob_dog(img, max_sigma=max_sigma, threshold=threshold)
        case _: 
            print("Algorithm not recognized.")
            sys.exit(1)

def img_blob_detection(img, algorithm, max_sigma, threshold):
    blobs_center = apply_algo(img, algorithm, max_sigma, threshold)
    mask = center_mask(blobs_center, img.shape)
    masked_img = np.stack([img, mask], axis=1)
    return [blobs_center, masked_img]

def blob_detection(input, output, algorithm, max_sigma, threshold, C, T):
    start_time = time.time()

    vol = tifffile.memmap(input)

    img = vol[:,:,C]

    composite = None
    columns_name = None
    center_coords = None
    if T is not None:
        columns_name = ["Z","Y","X","R"]
        [center_coords,composite] = img_blob_detection(img[T], algorithm, max_sigma, threshold)
    else:
        columns_name = ["T","Z","Y","X","R"]
        [t,z,c,y,x] = vol.shape
        composite = np.empty((t,z,2,y,x), dtype = 'uint8')
        center_coords = []
        for ti in tqdm(range(t)):
            [center_coord,composite[ti]] = img_blob_detection(img[ti], algorithm, max_sigma, threshold)
            # Add the t frame value as the first element of the array 
            center_coord = np.column_stack([np.full(len(center_coord), ti), center_coord])
            center_coords.append(center_coord)

    if len(center_coords) < 0:  # avoids error if empty
        print("No blob detected")
        sys.exit(1)
    elif T is None:
        center_coords = np.concatenate(center_coords, axis=0)

    print("Detection processing finished")
    output =  output
    os.makedirs(output, exist_ok=True)

    output_img = output / f"C{C}_detected_img.tif"
    tifffile.imwrite(output_img, 
                        composite,
                        imagej=True
                        )
    output_coords = output / f"C{C}_center_coord.csv"
    pd.DataFrame(center_coords, columns = columns_name).to_csv(output_coords)

    output_param = output / f"C{C}_params.csv"
    pd.DataFrame(
        [[str(input), C, T, algorithm,  max_sigma, threshold]],
        columns=["Input path", "Channel", "Time", "Algorithm", "Max_sigma", "Threshold"]
    ).to_csv(output_param)
    print(f"{algorithm} algorithm took {(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Difference of Gaussian algorithm to detect blob centers"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--algorithm", required=True, type=str)
    parser.add_argument("--max_sigma", required=True, type=float)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--channelID", required=True, type = int)
    parser.add_argument("--time", type = int)
    args = parser.parse_args()

    blob_detection(args.input_path, args.output_path, args.algorithm, args.max_sigma, args.threshold, args.channelID, args.time)

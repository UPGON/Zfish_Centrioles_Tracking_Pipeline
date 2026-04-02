import sys
import os 
import argparse
import pathlib
import time
import tifffile
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log
from tqdm import tqdm

def create_center_mask(blobs, shape, radius = 4):
    """Create a mask with circles at blob centers.
    
    Args:
        blobs: Array of blob coordinates (z, y, x, r)
        shape: Shape of the output mask (z, y, x)
        radius: Radius of circles to draw
    
    Returns:
        3D mask array with circles at blob centers
    """
    mask = np.zeros(shape,dtype=np.uint8)
    for z, y, x, *_ in blobs:
        z, y, x = int(z), int(y), int(x)
        cv2.circle(mask[z], center=(x,y), radius= radius, color=200, thickness=1)
    return mask

def apply_detection_algorithm(img,algorithm, max_sigma, threshold):
    """Apply blob detection algorithm.
    
    Args:
        img: Input 3D image (Z, Y, X)
        algorithm: 'dog' or 'log'
        max_sigma: Maximum sigma for Gaussian kernel
        threshold: Detection threshold
    
    Returns:
        Array of detected blobs (z, y, x, r)
    """
    algorithms = {
        'log': blob_log,
        'dog': blob_dog
    }

    if algorithm not in algorithms:
        raise ValueError(f"Algorithm '{algorithm}' not recognized. Use 'dog' or 'log'.")
    
    return algorithms[algorithm](img, max_sigma=max_sigma, threshold=threshold)

def frame_blob_detection(frame, algorithm, max_sigma, threshold):
    """Detect blobs and create an annotated images of the given 3D frame (image at a fixed time)
    
    Args:
        frame: 3D image (Z, Y, X)
        algorithm: Detection algorithm
        max_sigma: Maximum sigma
        threshold: Detection threshold
    
    Returns:
        Tuple of (blob_coordinates, composite)
    """
    blobs_center = apply_detection_algorithm(frame, algorithm, max_sigma, threshold)
    mask = create_center_mask(blobs_center, frame.shape)
    composite = np.stack([frame, mask], axis=1)
    return [blobs_center, composite]

def save_results(composite, blobs_coords,input_path, output_path, channel_id, algorithm,  max_sigma, threshold, timepoint):
    """ Save the composite image, the blobs centers coordinate and the parameters used
    
    Args:
        composite: Image made of the orignal image and mask with the blobs center marked
        blobs_coords: Array of the blobs coordinates
        input_path: Path to input TIFF
        output_path: Output directory
        channel_id: Channel to process
        algorithm: 'dog' or 'log'
        max_sigma: Maximum sigma for detection
        threshold: Detection threshold
        timepoint: Optional specific timepoint to process
    """
    print("Saving the restults")
    os.makedirs(output_path, exist_ok=True)

    #Save the annotated image
    output_img = output_path / f"C{channel_id}_detected_img.tif"
    tifffile.imwrite(output_img, 
                        composite,
                        imagej=True,
                        metadata={'axes': 'TZCYX' if timepoint is None else 'ZCYX'},
                        compression='zlib'
                        )
    # Save the blobs center coordinate
    output_coords = output_path / f"C{channel_id}_center_coord.csv"
    columns_name =  ["T","Z","Y","X","R"] if timepoint is None else ["Z","Y","X","R"]
    pd.DataFrame(blobs_coords, columns = columns_name).to_csv(output_coords, index_label ="index")

    # Save the parameters used for the detection
    output_param = output_path / f"C{channel_id}_params.csv"
    pd.DataFrame([{
        "Input path": str(input_path),
        "Channel": channel_id,
        "Timepoint": timepoint if timepoint is not None else "all",
        "Algorithm": algorithm,
        "Max_sigma": max_sigma,
        "Threshold": threshold,
        "Blobs detected": len(blobs_coords)
    }]).to_csv(output_param, index_label ="index")

def blob_detection(input_path, output_path, channel_id, algorithm, max_sigma, threshold, timepoint):
    """Main blob detection pipeline.
    
    Args:
        input_path: Path to input TIFF
        output_path: Output directory
        channel_id: Channel to process
        algorithm: 'dog' or 'log'
        max_sigma: Maximum sigma for detection
        threshold: Detection threshold
        timepoint: Optional specific timepoint to process
    """
    start_time = time.time()
    print("Loading the images")
    vol = tifffile.memmap(input_path)

    img = vol[:,:,channel_id]

    composite = None
    blobs_coord = None

    # Single frame (timepoint) to process
    if timepoint is not None:
        print(f"Processing a single frame at T={timepoint}")
        [blobs_coord,composite] = frame_blob_detection(img[timepoint], algorithm, max_sigma, threshold)
    # Process for all the frames 
    else:
        print(f"Processing all the frames")
        [t,z,y,x] = img.shape
        composite = np.empty((t,z,2,y,x), dtype = np.uint8)
        blobs_coords = []

        for ti in tqdm(range(t), desc="Detecting blobs in frames", unit="frame"):
            [blobs_coord,composite[ti]] = frame_blob_detection(img[ti], algorithm, max_sigma, threshold)
            
            # Add the t frame value as the first element of the array 
            blobs_coord = np.column_stack([np.full(len(blobs_coord), ti), blobs_coord])

            blobs_coords.append(blobs_coord)

        blobs_coords = np.concatenate(blobs_coords, axis=0) if blobs_coords else np.array([])


    if len(blobs_coord) == 0:  # avoids error if empty
        print("No blob detected")
        return

    save_results(composite, blobs_coords,input_path, output_path, channel_id, algorithm,  max_sigma, threshold, timepoint)

    print(f"{algorithm} algorithm took {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    """ Command-line interface for blob detection using Difference of Gaussian or Laplacian of Gaussian algorithms.
 
    Usage:
        python blob_detection.py --input_path <path_to_input_image> --output_path <path_to_output_directory> --channel_id <channel_id> --algorithm <'dog' or 'log'> --max_sigma <max_sigma_value> --threshold <threshold_value> [--timepoint <timepoint_to_process>]

    Args:
        --input_path (str): The path of the image to be processed. The image must be in 3D with format (T,Z,Y,X).
        --output_path (str): The path where the results should be saved (must be a directory).
        --channel_id (int): The channel to process.
        --algorithm (str): The blob detection algorithm to use ('dog' for Difference of Gaussian or 'log' for Laplacian of Gaussian).
        --max_sigma (float): The maximum sigma for the Gaussian kernel used in blob detection.
        --threshold (float): The detection threshold for blob detection.
        --timepoint (int, optional): If specified, only this timepoint will be processed. If not provided, all timepoints will be processed.
    """
    parser = argparse.ArgumentParser(
        description="Apply Difference of Gaussian algorithm to detect blob centers"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--channel_id", required=True, type = int)
    parser.add_argument("--algorithm", required=True, type=str)
    parser.add_argument("--max_sigma", required=True, type=float)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--timepoint", type = int)
    args = parser.parse_args()

    try:
        blob_detection(
            args.input_path, 
            args.output_path, 
            args.channel_id, 
            args.algorithm, 
            args.max_sigma, 
            args.threshold, 
            args.timepoint
        )
    except Exception as e:
        print(f"Error detected: {e}", file=sys.stderr)
        sys.exit(1)

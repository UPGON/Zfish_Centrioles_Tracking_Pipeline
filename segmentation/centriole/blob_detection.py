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
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects
from tqdm import tqdm

def save_results(composite, blobs_coords,input_path, output_path, channel_id, algorithm, min_sigma, max_sigma, threshold,radius_max,area_max, timepoint, z_min, z_max):
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
    output_img_path = output_path / f"C{channel_id}_detected_img.tif"
    tifffile.imwrite(output_img_path,
                    composite,
                    imagej=True,
                    metadata={'axes': 'TZCYX' if timepoint is None else 'ZCYX'},
                    compression='zlib'
                    )
    # Save the blobs center coordinate
    output_coords_path = output_path / f"C{channel_id}_center_coord.csv"
    columns_name =  ["T","Z","Y","X","R"] if timepoint is None else ["Z","Y","X","R"]
    pd.DataFrame(blobs_coords, columns = columns_name).to_csv(output_coords_path, index_label ="index")

    # Save the parameters used for the detection
    output_param_path = output_path / f"C{channel_id}_params.csv"
    pd.DataFrame([{
        "Blobs detected": len(blobs_coords),
        "Input path": str(input_path),
        "Channel": channel_id,
        "Timepoint": timepoint if timepoint is not None else "all",
        "Algorithm": algorithm,
        "Min_sigma": min_sigma,
        "Max_sigma": max_sigma,
        "Threshold": threshold,
        "Radius max":radius_max,
        "Area max": area_max,
        "Z min": z_min if z_min is not None else "all",
        "Z max": z_max if z_max is not None else "all"
    }]).to_csv(output_param_path, index_label ="index")

def verify_input(vol, channel_id, timepoint, z_min, z_max):
    """Verify the input parameters and volume dimensions.
    
    Args:
        vol: Input volume
        channel_id: Channel to process
        timepoint: Optional specific timepoint to process
        z_min: Minimum z slice to process
        z_max: Maximum z slice to process
    """
    if vol.ndim != 5:
        raise ValueError(f"Input volume must be 5D (T,Z,C,Y,X), but got shape {vol.shape}")
    
    T, Z, C, Y, X = vol.shape

    if channel_id < 0 or channel_id >= C:
        raise ValueError(f"Channel ID {channel_id} is out of bounds for volume with {C} channels.")
    
    if timepoint is not None and (timepoint < 0 or timepoint >= T):
        raise ValueError(f"Timepoint {timepoint} is out of bounds for volume with {T} timepoints.")
    
    if z_min is not None and (z_min < 0 or z_min >= Z):
        raise ValueError(f"z_min {z_min} is out of bounds for volume with {Z} z slices.")
    
    if z_max is not None and (z_max <= 0 or z_max > Z):
        raise ValueError(f"z_max {z_max} is out of bounds for volume with {Z} z slices.")
    
    if z_min is not None and z_max is not None and z_min >= z_max:
        raise ValueError(f"z_min {z_min} must be less than z_max {z_max}.")

def create_center_mask(blobs_coords, shape, radius = 4):
    """Create a mask with circles at blob centers.
    
    Args:
        blobs_coords: Array of blob coordinates (z, y, x, r)
        shape: Shape of the output mask (z, y, x)
        radius: Radius of circles to draw
    
    Returns:
        3D mask array with circles at blob centers
    """
    mask = np.zeros(shape,dtype=np.uint8)
    for z, y, x, *_ in blobs_coords:
        z, y, x = int(z), int(y), int(x)
        cv2.circle(mask[z], center=(x,y), radius= radius, color=200, thickness=1)
    return mask

def apply_detection_algorithm(img,algorithm,min_sigma, max_sigma, threshold):
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
    
    return algorithms[algorithm](img,min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)

def remove_large_area_element(frame,blob_center,area_max):
    # Take all the unique zframe
    zframes_detected = np.unique(blob_center[:,0]).astype(int)
    curated_blob_center = blob_center.copy()

    keep_mask = np.ones(len(blob_center), dtype=bool)

    for z in zframes_detected:

        thresh_yen = threshold_yen(frame[z])
        thresh_yen_img = frame[z] > thresh_yen
        img_labels = label(thresh_yen_img)
        large_obj = remove_small_objects(img_labels, min_size=area_max) != 0
        z_indices = np.where(blob_center[:,0] == z)[0]
        
        for idx in z_indices:
            y, x = blob_center[idx][1:3].astype(int)

            if(large_obj[y,x]):
                keep_mask[idx] = False  
    return curated_blob_center[keep_mask]
    

def frame_blob_detection(frame, algorithm,min_sigma, max_sigma, threshold,radius_max, area_max):
    """Detect blobs and create an annotated images of the given 3D frame (image at a fixed time)
    
    Args:
        frame: 3D image (Z, Y, X)
        algorithm: Detection algorithm
        max_sigma: Maximum sigma
        threshold: Detection threshold
    
    Returns:
        Tuple of (blob_coordinates, composite)
    """
    blobs_center = apply_detection_algorithm(frame, algorithm,min_sigma, max_sigma, threshold)

    if radius_max is not None:
        blobs_center = blobs_center[blobs_center[:,-1] <= radius_max]

    if area_max is not None: 
        blobs_center = remove_large_area_element(frame, blobs_center, area_max)

    mask = create_center_mask(blobs_center, frame.shape)
    composite = np.stack([frame, mask], axis=1)
    return [blobs_center, composite]

def blob_detection(input_path, output_path, channel_id, algorithm, min_sigma, max_sigma, threshold, radius_max = None, area_max = None, timepoint = None, z_min = None, z_max = None):
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
    verify_input(vol, channel_id, timepoint, z_min, z_max)
    [t,z,c,y,x] = vol.shape

    # If no z min and max were specified, then we process over the whole z stack
    z_min = z_min if z_min is not None else 0
    z_max = z_max if z_max is not None else z
    z_range = range(z_min, z_max)
    vol_c = vol[:,z_range,channel_id]

    composite = None
    blobs_coords = []

    # Single frame (timepoint) to process
    if timepoint is not None:
        print(f"Processing a single frame at T={timepoint}")
        [blobs_coords,composite] = frame_blob_detection(vol_c[timepoint-1], algorithm,min_sigma, max_sigma, threshold,radius_max,area_max)
    # Process for all the frames 
    else:
        print(f"Processing all the frames")
        
        composite = np.empty((t,len(z_range),2,y,x), dtype = np.uint8)

        for ti in tqdm(range(t), desc="Detecting blobs in frames", unit="frame"):
            [blobs_coord,composite[ti]] = frame_blob_detection(vol_c[ti], algorithm, min_sigma,max_sigma, threshold,radius_max,area_max)
            
            # Add the t frame value as the first element of the array 
            blobs_coord = np.column_stack([np.full(len(blobs_coord), ti), blobs_coord])

            blobs_coords.append(blobs_coord)

        blobs_coords = np.concatenate(blobs_coords, axis=0) if blobs_coords else np.array([])


    if len(blobs_coords) == 0:  # avoids error if empty
        print("No blob detected")
        return
    
    save_results(composite, blobs_coords,input_path, output_path, channel_id, algorithm, min_sigma,  max_sigma, threshold, radius_max,area_max, timepoint, z_min, z_max)

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
        description="Apply Scikit blob detection algorithm to detect blob centers"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--channel_id", required=True, type = int)
    parser.add_argument("--algorithm", required=True, type=str)
    parser.add_argument("--min_sigma", required=True, type=float)
    parser.add_argument("--max_sigma", required=True, type=float)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--radius_max", type=float)
    parser.add_argument("--area_max", type = float)
    parser.add_argument("--timepoint", type = int)
    parser.add_argument("--z_min", type = int)
    parser.add_argument("--z_max", type = int)
    args = parser.parse_args()

    try:
        blob_detection(
            args.input_path, 
            args.output_path, 
            args.channel_id, 
            args.algorithm,
            args.min_sigma, 
            args.max_sigma, 
            args.threshold, 
            args.radius_max,
            args.area_max,
            args.timepoint,
            args.z_min,
            args.z_max
        )
    except Exception as e:
        print(f"Error detected: {e}", file=sys.stderr)
        sys.exit(1)

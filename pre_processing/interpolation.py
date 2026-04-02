import argparse
import pathlib
import time
import tifffile
import cv2
import numpy as np
from itertools import product
from tqdm import tqdm


def interpolate(vol, scale, fx, fy):
    """Interpolate a 5D volume to a smaller size.
    
    Args:
        vol: Input volume with shape (T, Z, C, Y, X)
        scale: Scaling factor (0.75 = 75% of original size)
    
    Returns:
        Interpolated volume
    """
    T, Z, C, Y, X = vol.shape
    new_Y = int(Y * scale)
    new_X = int(X * scale)
    new_shape = (T, Z, C, new_Y, new_X)
    stack = np.empty(new_shape, vol.dtype)
    total = T*Z*C
    with tqdm(total=total, desc="Interpolating frames", unit="frame") as pbar:
        for t, z, c in product(range(T), range(Z), range(C)):
            img = vol[t,z,c]
            stack[t,z,c] = cv2.resize(
                        src = img,
                        dsize=((new_Y,new_X)),
                        fx = fx, 
                        fy = fy,
                        interpolation=cv2.INTER_LINEAR,
                    )
            pbar.update(1)
    return stack

def interpolation(input, output, scale, fx, fy):
    """ Handle the interpolation pipeline

    """
    start_time = time.time()
    memmap_volume = tifffile.memmap(input)
    interpolate_img = interpolate(memmap_volume, scale, fx, fy)
    tifffile.imwrite(output, 
                    interpolate_img,
                    imagej=True
                    )
    print(f"Interpolation took {(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    """ Command-line interface for interpolating a 5D volume using linear interpolation.
    
    Usage: 
        python interpolation.py --input_path <path_to_input_image> --output_path <path_to_output_image> --scale <scaling_factor> [--fx <fx>] [--fy <fy>]

    Args:
        --input_path (str): The path of the image to be interpolated. The image must be in 3D with format (T,Z,Y,X).
        --output_path (str): The path where the interpolated image(s) should be saved (must be a directory).
        --scale (float): The scaling factor for interpolation (e.g., 0.75 for 75% of original size).
        --fx (int, optional): The scaling factor along the x-axis. Default is 1.
        --fy (int, optional): The scaling factor along the y-axis. Default is 1.
    """
    parser = argparse.ArgumentParser(
        description="Interpolate the gien image using linear interpolation"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--fx", nargs='?', type=int, const=1)
    parser.add_argument("--fy", nargs='?', type=int, const=1)
    args = parser.parse_args()

    interpolation(args.input_path, args.output_path, args.scale, args.fx, args.fy)

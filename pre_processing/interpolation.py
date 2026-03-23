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
    parser = argparse.ArgumentParser(
        description="Crop the given image using the provided coordinates"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--fx", nargs='?', type=int, const=1)
    parser.add_argument("--fy", nargs='?', type=int, const=1)
    args = parser.parse_args()

    interpolation(args.input_path, args.output_path, args.scale, args.fx, args.fy)

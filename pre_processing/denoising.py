import argparse
import pathlib
import time
import tifffile
import cv2
import numpy as np
from itertools import product
from tqdm import tqdm
import skimage as ski
import sys
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
)
from scipy.ndimage import gaussian_filter

def denoise(vol, disk_size):
    """Denoise a 5D volume to a smaller size.
    
    Args:
        vol: Input volume with shape (T, Z, C, Y, X)
        scale: Scaling factor (0.75 = 75% of original size)
    
    Returns:
        Interpolated volume
    """
    T, Z, C, Y, X = vol.shape
    stack = np.empty(vol.shape, vol.dtype)
    total = T*Z*C
    with tqdm(total=total, desc="Denoising frames", unit="frame") as pbar:
        for t, z, c in product(range(T), range(Z), range(C)):
            img = vol[t,z,c]

            # Background subtraction
            background = gaussian_filter(img.astype(float), sigma=20)
            corrected = img.astype(float) - background
            corrected = np.clip(corrected, 0, None)

            footprint = ski.morphology.disk(disk_size)
            """ 
            filtered =  ski.filters.rank.median(
                            img,
                            footprint
                        )
                        """
            stack[t,z,c] = ski.morphology.white_tophat(img.astype(np.uint8), footprint)

            """        
            stack[t,z,c] = ski.filters.rank.median(
                            img,
                            footprint
                        )"""
                    
            pbar.update(1)
    return stack

def gaussian_denoising(vol):
    T, Z, C, Y, X = vol.shape
    stack = np.empty(vol.shape, vol.dtype)
    total = T*Z*C
    with tqdm(total=total, desc="Denoising frames", unit="frame") as pbar:
        for t, c in product(range(T), range(C-1)):
            img = vol[t,:,c]

            background = gaussian_filter(img.astype(float), sigma=20)
            corrected = img.astype(float) - background
            corrected = np.clip(corrected, 0, None)

            stack[t,:,c] = gaussian_filter(
                            corrected,
                            sigma=1
                        )
            pbar.update(1)
    return stack


def denoising(input, output, disk_size):
    """ Handle the denoising pipeline

    """
    start_time = time.time()
    memmap_volume = tifffile.memmap(input)
    denoised_img = denoise(memmap_volume,disk_size)
    tifffile.imwrite(output, 
                    denoised_img,
                    imagej=True
                    )
    print(f"Denoising took {(time.time() - start_time):.2f} seconds")

if __name__ == "__main__":
    """ Command-line interface for denoising a 5D volume using median filter.
    
    Usage: 
        python denoising.py --input_path <path_to_input_image> --output_path <path_to_output_image> --disk_size <disk_size>

    Args:
        --input_path (str): The path of the image to be interpolated. The image must be in 3D with format (T,Z,Y,X).
        --output_path (str): The path where the interpolated image(s) should be saved (must be a directory).
        --disk_size (float): 
    """
    parser = argparse.ArgumentParser(
        description="Interpolate the gien image using linear interpolation"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--disk_size", required=True, type=float)
    args = parser.parse_args()

    denoising(args.input_path, args.output_path, args.disk_size)

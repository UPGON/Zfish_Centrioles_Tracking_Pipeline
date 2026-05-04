import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def verif_stack_point(stack, point):
    z, y, x = point.astype(int)

    z_max, y_max, x_max = stack.shape
    if z < 0 or z > z_max:
        raise ValueError(f"z coordinates of the point is out of bound for this stack")
    if y < 0 or y >= y_max:
        raise ValueError(f"y coordinate {y} is out of bounds for stack height {y_max}.")
    if x < 0 or x >= x_max:
        raise ValueError(f"x coordinate {x} is out of bounds for stack width {x_max}.")


def clamping_window_size(window_size, idx, img_size):
    idx_min = np.clip(idx - window_size//2, 0, img_size)
    idx_max = np.clip(idx + window_size//2, 0, img_size)

    if(idx_min == 0):
        idx_max = window_size
    if(idx_max == img_size):
        idx_min = img_size - window_size
    return idx_min, idx_max


def get_window_range(window_size, point, size):
    window_range = []
    for i in range(len(point)):
        coord = point[i].astype(int)        
        idx_min, idx_max = clamping_window_size(window_size, coord, size[i])
        window_range.append(slice(idx_min, idx_max))

    return tuple(window_range)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
    

def center_detection_annotation(img, center_coord, radius = 8, color = 255, thickness = 1):
    res_img = img.copy()
    for z, y, x in center_coord:
        z, y, x = int(z), int(y), int(x)
        cv2.circle(res_img[z], center=(x,y), radius= radius, color=color, thickness=thickness)
    return res_img

def annotate_pairing_mask(text, idx, center_df, mask, font = cv2.FONT_HERSHEY_COMPLEX_SMALL):
    (x, y, z) = center_df[["X","Y","Z"]].loc[idx].values

    y_max, x_max = mask.shape[-2:]
    x_offset = 2
    y_offset = 5
    xy_offset = (int(np.clip(x + x_offset,0, x_max)), int(np.clip(y - y_offset, 0, y_max)))

    cv2.putText(img = mask[int(z)],text= text,org= xy_offset,fontFace= font,fontScale= 0.4,color = (255),thickness= 1,lineType= cv2.LINE_AA)

def plot_comparison_spots_detection(coord1, coord2, label1, label2, img, window_size = 40):
    verif = img.copy()

    cv2.circle(verif, (coord1[0], coord1[1]), radius=4, color=200, thickness=1)
    cv2.putText(verif, label1, (coord1[0] + 5, coord1[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 200, 1,lineType= cv2.LINE_AA)
    cv2.circle(verif, (coord2[0], coord2[1]), radius=4, color=255, thickness=1, lineType=8)
    cv2.putText(verif, label2, (coord2[0] + 5, coord2[1] + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 255, lineType= cv2.LINE_AA)

    window_slice = (slice(coord1[1] - window_size, coord1[1] + window_size), slice(coord1[0] - window_size, coord1[0] + window_size))

    fig, axes = plt.subplots(1,2, figsize=(20,8))
    axes[0].imshow(img[window_slice], cmap="inferno")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(verif[window_slice], cmap="inferno")
    axes[1].set_title(f"Annotated image with {label1}  and {label2} detections")
    axes[1].axis("off")

    fig.suptitle("Spots detection verfication of edge case")

def plot_zoomed_histogram(values, zooming_threshold, threshold, title, unit, below = True, bins = 100):
    fig, axes = plt.subplots(1,2, figsize=(12,4))

    axes[0].hist(values, bins=bins)
    axes[0].set_xlabel(unit)
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0,values.max() + 10)
    axes[0].axvline(0, color = 'g', linestyle = 'dashed', linewidth = 2)
    axes[0].axvline(zooming_threshold, color = 'g', linestyle = 'dashed', linewidth = 2)
    axes[0].legend(["Zoomed regions"])

    if below: 
        axes[1].hist(values[values < zooming_threshold], bins=bins//2)
    else:
        axes[1].hist(values[values > zooming_threshold], bins=bins//2)
    axes[1].set_xlabel(unit)
    axes[1].set_ylabel("Frequency")
    if below: 
        axes[1].set_xlim(0,zooming_threshold + 10)
    else:
        axes[1].set_xlim(zooming_threshold + 10,values.max() + 10)
    axes[1].axvline(threshold, color = 'r', linestyle = 'dashed', linewidth = 2)
    axes[1].legend(["Threshold"])
    fig.suptitle(title)
    plt.show()

def ensure_window_limit(window_size, idx, img_size):
    idx_min = min(idx, window_size)
    idx_max = min(img_size - idx, window_size)
    return min(idx_min, idx_max)
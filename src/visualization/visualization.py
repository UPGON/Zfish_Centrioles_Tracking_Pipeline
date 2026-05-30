import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils 

# Maximal distance between any label center and the image border in pixel
LABEL_MARGIN = 5

def draw_circles2D(img, centers, radius=[4], color=255, thickness=1):
    """Draw circles on the stack images at the centers coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    if(len(radius) == 1):
        r = np.full(len(centers), radius)
    elif(len(radius) != len(centers)):
        raise ValueError("The number of radius should be either 1 or the same as the number of centers")
    res_img = img.copy()
    for i in range(len(centers)):
        center = centers[i]
        y, x = center.astype(int)
        ri = r[i] if len(radius) > 1 else r[0]
        utils.verif_img_point(img, center)
        cv2.circle(res_img, center=(x, y), radius=ri*2, color=color, thickness=thickness)
    return res_img

def draw_circles(stack, centers, radius=[4], color=255, thickness=1):
    """Draw circles on the stack images at the centers coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    if(len(radius) == 1):
        radius = np.full(len(centers), radius)
    elif(len(radius) != len(centers)):
        raise ValueError("The number of radius should be either 1 or the same as the number of centers")
    res_stack = stack.copy()
    for i in range(len(centers)):
        center = centers[i]
        z, y, x = center.astype(int)
        ri = radius[i].astype(int)
        utils.verif_stack_point(stack, center)
        cv2.circle(res_stack[z], center=(x, y), radius=ri*2, color=color, thickness=thickness)
    return res_stack

def create_circle_mask(centers, vol, radius=[4], color = 200, thickness = 1):
    """Create a mask with circles 

    Args:
        blobs_coords: Array of blob coordinates (z, y, x)
        shape: Shape of the output mask (z, y, x)
        radius: Radius of circles to draw

    Returns:
        3D mask array with circles at blob centers
    """
    if(len(radius) == 1):
        radius = np.full(len(centers), radius)
    elif(len(radius) != len(centers)):
        raise ValueError("The number of radius should be either 1 or the same as the number of centers")
    
    mask = np.zeros(vol.shape, dtype=vol.dtype)
    for i in range(len(centers)):
        center = centers[i]
        utils.verif_stack_point(mask, center)
        z, y, x = center.astype(int)
        ri = radius[i].astype(int)
        cv2.circle(mask[z], center=(x, y), radius=ri*2, color=200, thickness=1)
    return mask


def add_text(stack,text, coord, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, text_offset = (-5,2), color = 255):
    """Add text annotation on the stack images at the given coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    res_stack = stack.copy()
    utils.verif_stack_point(stack, coord)
    y_size, x_size = res_stack.shape[-2:]

    text_coord = np.clip(coord[-2:] + text_offset, (LABEL_MARGIN,LABEL_MARGIN), (y_size-LABEL_MARGIN, x_size-LABEL_MARGIN)).astype(int)

    cv2.putText(
        img=res_stack[int(coord[0])],
        text=text,
        org=np.flip(text_coord),
        fontFace=font,
        fontScale=0.4,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return res_stack

def add_texts(stack,texts,coords, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, text_offset = (-5,2), color = 255):
    """Add text annotation on the stack images at the given coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        texts: 
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    res_stack = stack.copy()
    for text, coord in zip(texts, coords):
        res_stack = add_text(res_stack,text, coord, font=font, text_offset=text_offset, color=color)
    
    return res_stack

def plot_points_comparison(stack, points, labels, window_size=60, z_margin=4, text_offset=(-5, 5)):
    """Plot a close-up view of the z-max projection stack for multiple points."""
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length.")
    if len(points) == 0:
        raise ValueError("points cannot be empty.")

    for point in points:
        if point.size != 3:
            raise ValueError("Each point must be a 3D [z,y,x] coordinate.")
        utils.verif_stack_point(stack, point)

    annotated_stack_slice = stack.copy()
    annotated_stack_slice = draw_circles(annotated_stack_slice, points)

    for point, label in zip(points, labels):
        _, y, x = point
        annotated_stack_slice = add_text(annotated_stack_slice, label,point, text_offset=text_offset, color = 200)

    z_size = stack.shape[0]
    z_coords = [point[0] for point in points]
    z_min = min(z_coords)
    z_max = max(z_coords)
    z_range = slice(max(0, z_min - z_margin), min(z_size, z_max + z_margin))

    zmax_stack_slice = np.max(annotated_stack_slice[z_range], axis=0)

    midpoint = np.mean(points, axis=0).astype(int)

    window_range = utils.get_window_range(window_size, midpoint[-2:], zmax_stack_slice.shape)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(zmax_stack_slice[window_range], cmap="inferno")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    zmax_annotated_stack_slice = np.max(annotated_stack_slice, axis =0)
    axes[1].imshow(zmax_annotated_stack_slice[window_range], cmap="inferno")
    axes[1].set_title(f"Annotated image with {', '.join(labels)}")
    axes[1].axis("off")

    fig.suptitle("Spots detection verification of edge case")
    plt.show()

def plot_points_closeup(stack, points,radius, window_size = 46, z_margin=4):
    if len(points) == 0:
        raise ValueError("points cannot be empty.")

    for point in points:
        if point.size != 3:
            raise ValueError("Each point must be a 3D [z,y,x] coordinate.")
        utils.verif_stack_point(stack, point)

    n_plots = len(points)
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 8))
    for i in range(n_plots):
        z,y,x = points[i]

        z_size = stack.shape[0]
        z_range = slice(max(0, z - z_margin), min(z_size, z + z_margin))
        window_range = utils.get_window_range(window_size, [y,x],stack.shape[1:3])

        zmax_stack = np.max(stack[z_range], axis=0)
        axes[i].imshow(zmax_stack[window_range], cmap="inferno")
        axes[i].set_title(f"Rad: {radius[i]:.02f}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_zoomed_histogram(values, zooming_threshold, threshold, title, unit, below=True, bins=100,output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(values, bins=bins)
    axes[0].set_xlabel(unit)
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, values.max())
    axes[0].axvline(0, color="g", linestyle="dashed", linewidth=2)
    axes[0].axvline(zooming_threshold, color="g", linestyle="dashed", linewidth=2)
    axes[0].legend(["Zoomed regions"])

    if below:
        axes[1].hist(values[values < zooming_threshold], bins=bins // 2)
    else:
        axes[1].hist(values[values > zooming_threshold], bins=bins // 2)
    axes[1].set_xlabel(unit)
    axes[1].set_ylabel("Frequency")
    if below:
        axes[1].set_xlim(0, zooming_threshold)
    else:
        axes[1].set_xlim(zooming_threshold, values.max())
    axes[1].axvline(threshold, color="r", linestyle="dashed", linewidth=2)
    axes[1].legend(["Threshold"])
    fig.suptitle(title)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)

    if(show):
        plt.show()

def plot_proportions(values ,labels, title = "Proportion", colors =None,  output_path=None, show=True):
    if len(values) != len(labels):
        raise ValueError("Number of values must match number of labels")

    if colors is None:
        colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Proportion")
    ax.set_title(title)

    ax.bar_label(
        bars,
        labels=[f"{p*100:.1f}%" for p in values],
        fontsize=10
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)

    if(show):
        plt.show()

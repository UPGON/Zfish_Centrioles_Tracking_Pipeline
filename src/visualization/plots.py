
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils
from visualization import visualization

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
    annotated_stack_slice = visualization.draw_circles(annotated_stack_slice, points)

    for point, label in zip(points, labels):
        _, y, x = point
        annotated_stack_slice = visualization.add_text(annotated_stack_slice, label,point, text_offset=text_offset, color = 200)

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

def plot_points_closeup(stack, points,legends=None, window_size = 46, z_margin=4):
    if len(points) == 0:
        raise ValueError("points cannot be empty.")

    for point in points:
        if point.size != 3:
            raise ValueError("Each point must be a 3D [z,y,x] coordinate.")
        utils.verif_stack_point(stack, point)

    n_plots = len(points)
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 8))
    if n_plots == 1:
        axes = np.array([axes])

    for i in range(n_plots):
        z, y, x = points[i]

        z_size = stack.shape[0]
        z_range = slice(max(0, z - z_margin), min(z_size, z + z_margin))
        window_range = utils.get_window_range(window_size, [y, x], stack.shape[1:3])

        zmax_stack = np.max(stack[z_range], axis=0)
        axes[i].imshow(zmax_stack[window_range], cmap="inferno")
        if legends is not None:
            axes[i].set_title(f"Rad: {legends[i]:.02f}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_zoomed_histogram(values, zooming_threshold, threshold, title, unit, below=True, bins=100,output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(values, bins=bins)
    axes[0].set_xlabel(unit)
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, values.max()*1.2)
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

def plot_count(values ,labels, title = "Number of foci", colors =None,  output_path=None, show=True):
    if len(values) != len(labels):
        raise ValueError("Number of values must match number of labels")

    if colors is None:
        colors = plt.cm.tab10.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(4, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("# foci")
    ax.set_title(title)

    ax.bar_label(
        bars,
        labels=[f"{p} ({p/values.sum()*100:.1f}%)" for p in values],
        fontsize=10
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)

    if(show):
        plt.show()
        
def plot_comparison_box_plot(values, labels, title, unit, colors=None, linestyle=None, legends=None, output_path=None, show=True):
    fig, ax = plt.subplots(figsize=(6, 8))

    if colors is None:
        colors = plt.cm.tab10.colors[: len(labels)]

    if linestyle is None:
        line_styles = ["-"] * len(labels)
    elif isinstance(linestyle, (list, tuple)):
        if len(linestyle) != len(labels):
            raise ValueError("linestyle list must have the same length as labels")
        line_styles = list(linestyle)
    else:
        raise TypeError("linestyle must be None or a list/tuple of line styles")

    bp = ax.boxplot(values, labels=labels, patch_artist=True, medianprops={"color": "black"})

    for i, box in enumerate(bp["boxes"]):
        # Set facecolor with transparency, edgecolor with full opacity
        # Convert color name to RGBA, then set alpha to 0.2
        color_rgba = to_rgba(colors[i])
        facecolor_rgba = (*color_rgba[:3], 0.2)
        box.set_facecolor(facecolor_rgba)
        box.set_edgecolor("black")
        box.set_linestyle(line_styles[i])
        box.set_linewidth(2)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Overlay jittered scatter points for each box
    for i, vals in enumerate(values):
        vals_arr = np.asarray(vals)
        if vals_arr.size == 0:
            continue
        # jitter around box x-position (i+1)
        jitter = 0.04
        xs = np.random.normal(i + 1, jitter, size=len(vals_arr))
        scatter_color = colors[i] if colors is not None and i < len(colors) else "red"
        ax.scatter(xs, vals_arr, alpha=0.4, color=scatter_color, s=20, edgecolors="black", linewidths=0.3)

    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    if legends is not None:
        handles = []
        if isinstance(legends, dict):
            for legend_label, legend_style in legends.items():
                handles.append(
                    mlines.Line2D([], [], color="black", linestyle=legend_style, linewidth=2, label=legend_label)
                )
        else:
            raise TypeError("legends must be a dict")

        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
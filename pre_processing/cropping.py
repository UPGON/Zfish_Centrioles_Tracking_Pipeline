import sys
import argparse
import cv2
import pathlib

class ImageNotFoundError(Exception):
    pass


def open_image(image_path):
    """ Opens an image from a given path and returns it as a numpy array.

    Args:
        image_path (pathlib.Path): The path of the image to be opened.
    
    Returns:
        numpy.ndarray: The opened image as a numpy array.
    """
    img = cv2.imread(image_path)
    if img is None: 
        raise ImageNotFoundError(f"Image not found at path: {image_path}")
    else: 
        return img

def crop_image(image, x_start, y_start, x_end, y_end):
    """ Crops the given image at the provided coordinates.
    X-coordinates goes from left to right and Y-coordinates goes from top to bottom.

    Args: 
        image (numpy.ndarray): The image to be cropped.
        x_start (int): The x-coordinate of the top-left corner of the cropping rectangle.
        y_start (int): The y-coordinate of the top-left corner of the cropping rectangle.
        x_end (int): The x-coordinate of the bottom-right corner of the cropping rectangle.
        y_end (int): The y-coordinate of the bottom-right corner of the cropping rectangle.

    Returns: 
        numpy.ndarray: The cropped image.
    """
    img_height, img_width = image.shape[2:4]
    assert(0 <= x_start <= img_width, "x_start should be between the image size limits")
    assert(x_start < x_end, "x_start should be smaller than x_end")
    assert(0 <= y_start <= img_height, "y_start should be between the image size limits")
    assert(y_start < y_end, "y_start should be smaller than y_end")

    crop_img = image[:][:][y_start:y_end, x_start:x_end][:]
    return crop_img

def save_image(image, output_path):
    """ Saves the given image at the provided path.

    Args:
        image (numpy.ndarray): The image to be saved.
        output_path (pathlib.Path): The path where the image should be saved.
    """
    writeStatus = cv2.imwrite(output_path, image)
    if writeStatus:
        print(f"Image successfully saved at: {output_path}")
    else :
        print(f"Failed to save image at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop the given image using the provided coordinates"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    parser.add_argument("--x_start", required=True, type=int)
    parser.add_argument("--y_start", required=True, type=int)
    parser.add_argument("--x_end", required=True, type=int)
    parser.add_argument("--y_end", required=True, type=int)
    args = parser.parse_args()

    try:
        img = open_image(args.input_path)
        cropped_img = crop_image(
            img, 
            args.x_start,
            args.y_start,
            args.x_end,
            args.y_end
        )
        save_image(cropped_img, args.output_path)
    except ImageNotFoundError as e:
        print(e)
        sys.exit(1)
    except AssertionError as e:
        print(e)
        sys.exit(1)
import sys
import argparse
import pathlib
import multiprocessing
import time
import tifffile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def open_image(image_path):
    """ Opens an image from a given path and returns it as a numpy array.

    Args:
        image_path (pathlib.Path): The path of the image to be opened.
    
    Returns:
        numpy.ndarray: The opened image as a numpy array.
    """
    img = tifffile.imread(image_path)
    if img is None: 
        raise ValueError(f"Image not found at path: {image_path}")
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
    if (len(image.shape) != 3):
        raise ValueError(f"Expected a 3D image (Z, Y, X) but got an image with shape {image.shape}")

    img_height, img_width = image.shape[-2:]

    if not (0 <= x_start < x_end <= img_width):
        raise ValueError(f"x_start ({x_start}) must be >= 0, < x_end ({x_end}) and <= width ({img_width})")
    if not (0 <= y_start < y_end <= img_height):
        raise ValueError(f"y_start ({y_start}) must be >= 0, < y_end ({y_end}) and <= width ({img_width})")
    crop_img = image[:, y_start:y_end, x_start:x_end]
    return crop_img

def save_image(image, output_file):
    """ Saves the given image at the provided path.

    Args:
        image (numpy.ndarray): The image to be saved.
        output_path (pathlib.Path): The path where the image should be saved (must be a file path, not a directory).
    """
    if output_file.exists():
            print(f"Warning: Output file {output_file} already exists and will be overwritten.")
    tifffile.imwrite(output_file, image)

def crop_file(input_path, output_path, x_start, y_start, x_end, y_end):
    """ Crops the image at the given input path using the provided coordinates and saves it at the given output path.

    Args:
        input_path (pathlib.Path): The path of the image to be cropped.
        output_path (pathlib.Path): The path where the cropped image should be saved (must be a file path, not a directory).
        x_start (int): The x-coordinate of the top-left corner of the cropping rectangle.
        y_start (int): The y-coordinate of the top-left corner of the cropping rectangle.
        x_end (int): The x-coordinate of the bottom-right corner of the cropping rectangle.
        y_end (int): The y-coordinate of the bottom-right corner of the cropping rectangle.

    Returns:
        tuple: (success: bool, filename: str, message: str)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    img = open_image(input_path)
    cropped_img = crop_image(
        img, 
        x_start,
        y_start,
        x_end,
        y_end
    )
    output_file = output_path / input_path.name
    save_image(cropped_img, output_file)
    

def crop_file_task(args):
    """ Wrapper function for parallel processing. """
    return crop_file(*args)

def cropping(input_path, output_path, x_start, y_start, x_end, y_end): 
    """ Crops an image or a set of images in a directory based on provided command-line arguments.
       Args:
        input_path (pathlib.Path): The path of the image to be cropped.
        output_path (pathlib.Path): The path where the cropped image should be saved (must be a file path, not a directory).
        x_start (int): The x-coordinate of the top-left corner of the cropping rectangle.
        y_start (int): The y-coordinate of the top-left corner of the cropping rectangle.
        x_end (int): The x-coordinate of the bottom-right corner of the cropping rectangle.
        y_end (int): The y-coordinate of the bottom-right corner of the cropping rectangle.

    """
    if input_path.is_file():
            crop_file(input_path, output_path, x_start, y_start, x_end, y_end)
    elif input_path.is_dir():
        image_extensions = {".tif",".tiff",".png",".jpg",".jpeg"}

        images_path = [p for p in input_path.glob("*.*") if p.suffix.lower() in image_extensions]
        if not images_path:
            print("No images found in the provided directory.")
            sys.exit(0)

        max_workers = multiprocessing.cpu_count() - 1
        tasks = [(img_path, output_path, x_start, y_start, x_end, y_end) 
            for img_path in images_path]
        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            futures = {executor.submit(crop_file_task, task) for task in tasks}
        
            for future in tqdm(as_completed(futures), total=len(futures), 
                       desc="Cropping images", unit="img"):
                try: 
                    future.result()
                except (ValueError, OSError) as e:
                    print(f"Error processing {input_path.name}: {e}")
    else: 
        print(f"Error: {input_path} is neither a file nor a directory.")
        sys.exit(1)
    print(f"Successfully cropped images")
    print(f"Operation time: {time.time() - start:.6f}s")

if __name__ == "__main__":
    """ Main function to crop an image or a set of images in a directory based on provided command-line arguments.

    Usage:
        python cropping.py --input_path <path_to_image_or_directory> --output_path <path_to_output_directory> --x_start <x_start> --y_start <y_start> --x_end <x_end> --y_end <y_end>
    
    Args:
        --input_path (str): The path of the image to be cropped or a directory containing images to be cropped.
        --output_path (str): The path where the cropped image(s) should be saved (must be a directory).
        --x_start (int): The x-coordinate of the top-left corner of the cropping rectangle.
        --y_start (int): The y-coordinate of the top-left corner of the cropping rectangle.
        --x_end (int): The x-coordinate of the bottom-right corner of the cropping rectangle.
        --y_end (int): The y-coordinate of the bottom-right corner of the cropping rectangle.
    """
    start = time.time()
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

    cropping(args.input_path, args.output_path, args.x_start, args.y_start, args.x_end, args.y_end)

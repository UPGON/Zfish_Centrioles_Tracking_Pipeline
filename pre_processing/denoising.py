import sys
import argparse
import pathlib
import time
import tifffile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def crop_file(input_path, output_path, x_start, y_start, x_end, y_end):
    """ Crops the image at the given input path using the provided coordinates and saves it at the given output path.

    Args:
        input_path (pathlib.Path): The path of the image to be cropped.
        output_path (pathlib.Path): The path where the cropped image should be saved (must be a file path, not a directory).
        x_start (int): The x-coordinate of the top-left corner of the cropping rectangle.
        y_start (int): The y-coordinate of the top-left corner of the cropping rectangle.
        x_end (int): The x-coordinate of the bottom-right corner of the cropping rectangle.
        y_end (int): The y-coordinate of the bottom-right corner of the cropping rectangle.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    img = tifffile.imread(input_path)
    cropped_img = crop_image(
        img, 
        x_start,
        y_start,
        x_end,
        y_end
    )
    output_file = output_path / input_path.name
    if output_file.exists():
            print(f"Warning: Output file {output_file} already exists and will be overwritten.")
    tifffile.imwrite(output_file, cropped_img)
    
def crop_file_task(args):
    """ Wrapper function for parallel processing. """
    return crop_file(*args)

def crop_data_set(input_path, output_path, x_start, y_start, x_end, y_end):
    """ Crops all images in the given input directory using the provided coordinates and saves them at the given output directory.
    
    Args:
        input_path (pathlib.Path): The path of the image to be cropped.
        output_path (pathlib.Path): The path where the cropped image should be saved (must be a file path, not a directory).
        x_start (int): The x-coordinate of the top-left corner of the cropping rectangle.
        y_start (int): The y-coordinate of the top-left corner of the cropping rectangle.
        x_end (int): The x-coordinate of the bottom-right corner of the cropping rectangle.
        y_end (int): The y-coordinate of the bottom-right corner of the cropping rectangle.
    """
    image_extensions = {".tif",".tiff",".png",".jpg",".jpeg"}

    images_path = [p for p in input_path.glob("*.*") if p.suffix.lower() in image_extensions]
    if not images_path:
        print("No images found in the provided directory.")
        sys.exit(0)

    tasks = [(img_path, output_path, x_start, y_start, x_end, y_end) 
        for img_path in images_path]
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(crop_file_task, task) for task in tasks}
    
        for future in tqdm(as_completed(futures), total=len(futures), 
                desc="Cropping images", unit="img"):
            try: 
                future.result()
            except (ValueError, OSError) as e:
                print(f"Error processing {input_path.name}: {e}")

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
        crop_data_set(input_path, output_path, x_start, y_start, x_end, y_end)
    else: 
        print(f"Error: {input_path} is neither a file nor a directory.")
        sys.exit(1)
    print(f"Successfully cropped images")
    print(f"Operation time: {time.time() - start:.6f}s")

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Denoise the given image"
    )
    parser.add_argument("--input_path", required=True, type=pathlib.Path)
    parser.add_argument("--output_path", required=True, type=pathlib.Path)
    args = parser.parse_args()

    cropping(args.input_path, args.output_path, args.x_start, args.y_start, args.x_end, args.y_end)

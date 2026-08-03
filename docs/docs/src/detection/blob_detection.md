# Spot detection (blob_detection.py)
Detect bright spots on a dark background from a given image channel.

This script loads a TIFF image, extracts a selected channel, runs blob detection with either the `dog` or `log` algorithm, and optionally writes composite output images with annotations.
The area and intensity of each spot are also measured and saved in the results.

## Input parameters
Required

- ***input_path*** `[pathlib.Path]`: Path to the input TIFF file. 
- ***channel_id*** `[int]`: Channel index to process. The index must be greater than 0 (do not use coding index formatting).
- ***algorithm*** `["dog","log"]`: Detection algorithm. 
- ***threshold*** `[float]`: Detection threshold.

Optional
- ***output_path*** `[pathlib.Path]`: Output directory.
- ***min_radius*** `[float]`: Minimum radius in µm. Default: `0.2`.
- ***max_radius*** `[float]`: Maximum radius in µm. Default: `1.0`.
- ***area_max*** `[float]`: Maximum area in square micrometres for filtering.
- ***merging_distance*** `[float]`: Maximum distance used to merge nearby detected centres in µm.
- ***timepoint*** `[int]`: Index of the single frame to process.
- ***z_min*** `[int]`: Minimum Z slice to process.
- ***z_max*** `[int]`: Maximum Z slice to process.
- ***annotation*** `[bool]`: Whether to include index labels in the output images. Default: `True`.
- ***drawn_radius*** `[float]`: Radius used when drawing detected blobs in the output composite. Default: `None`.
- ***max_workers*** `[int]`: Number of parallel workers. Default: CPU count minus one.

## Output results  
- ***cX_centers.csv***: 3D (+t) coordinates and features (area & intensity) of the detected spots.
- ***cX_detection_img.tif***: TIFF images with the channel image, bright field and detected spot annotations.
- ***cX_params.csv***: parameters used for the detection.

## Command usage
```bash
@echo off
python path/blob_detection.py --input_path INPUT_PATH --output_path OUTPUT_PATH --channel_id CHANNEL_ID --algorithm ALGORITHM --threshold THRESHOLD --min_radius MIN_RADIUS --max_radius MAX_RADIUS --merging_distance MERGING_DISTANCE --timepoint TIMEPOINT --z_min Z_MIN --z_max Z_MAX --annotation ANNOTATION --drawn_radius DRAWN_RADIUS --max_workers MAX_WORKERS
```


### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\src\detection\blob_detection.py --input_path "..\..\..\subsets\20260617CFHb_CetnEos_H2BmCherry_LynTomato_CS_1dpf_TL.lif - 2e1-1.tif" --output_path "cs_res" --channel_id 3 --algorithm "dog" --threshold 0.05 --min_radius 0.3 --max_radius 1.4 --drawn_radius 3 --merging_distance 0.7 
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\detection\blob_detection.py --input_path "..\..\..\subsets\20260617CFHb_CetnEos_H2BmCherry_LynTomato_CS_1dpf_TL.lif - 2e1-1.tif" --output_path "cs_res" --channel_id 3 --algorithm "dog" --threshold 0.05 --min_radius 0.3 --max_radius 1.4 --merging_distance 0.7 --timepoint 1 --z_min 10 --z_min 20 --annotation False --drawn_radius 3  --max_workers 5
```


## Suggested folder structure 

```
experiment
├── images.tif            <- original, raw images from the experiment
├── subsets           
│   └── s1_images.tif           <- subset of the original images, can notably contain its cropped version. We usually run the detection on image subsets
└── results           
   └── detection
        └── raw 
            └── blob_detection_cX.bat        <- command to run the spot detection script
```
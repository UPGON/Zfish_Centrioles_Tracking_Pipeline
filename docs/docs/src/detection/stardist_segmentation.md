# Nuclei segmentation (stardist_segmentation.py)

Segment nuclei in a given channel using a pre-trained StarDist model.

This script loads a TIFF image, extracts the requested channel, and runs a StarDist-based segmentation workflow. It can process a single timepoint or the full stack, and supports custom model settings and resolution parameters.

## Input parameters
Required
- ***input_path*** `[Path]`: Path to the input TIFF file.
- ***output_path*** `[Path]`: Output directory where the segmentation results will be written.
- ***channel_id*** `[int]`: Channel index to process.
- ***model_resolution*** `[float float float]`: Resolution of the trained model images in `[µm/px]`, provided as `DZ DY DX`.

Optional
- ***model_path*** `[str]`: Path to a custom pre-trained StarDist model.
- ***proba_thresh*** `[float]`: Probability threshold used by the segmentation model.
- ***nms_thresh*** `[float]`: Non-maximum suppression threshold.
- ***blur*** `[bool]`: Whether to apply blur before segmentation. Use `True` or `False`.
- ***timepoint*** `[int]`: If provided, only this timepoint is processed.
- ***z_min*** `[int]`: Minimum Z slice to process.
- ***z_max*** `[int]`: Maximum Z slice to process.
- ***resolution_as_scale*** `[bool]`: Whether the provided resolution should be used as a scaling factor. Use `True` or `False`.

## Output results
- ***cX_centers.csv***: 3D (+t) coordinates and features (area & intensity) of the detected nuclei.
- ***cX_segmentation.tif***: TIFF image with segmented nuclei with centre and index annotations.
- ***cX_params.csv***: parameters used for the detection.

## Command usage

```bash
python src/detection/stardist_segmentation.py --input_path INPUT_PATH --output_path OUTPUT_PATH --channel_id CHANNEL_ID --model_resolution DZ DY DX
```


### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\src\detection\stardist_segmentation.py --input_path "..\..\..\subsets\s1_20260619CFHa_CetnEos_H2BmCherry_LynTomato_CS_3dpf.lif - 1e7-1.tif" --output_path "h2b_res" --channel_id 2 --model_resolution 1 0.7 0.4 --model_path ..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\models\stardist\generic_plant_nuclei_3D --resolution_as_scale True 
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\detection\stardist_segmentation.py --input_path "..\..\..\subsets\s1_20260619CFHa_CetnEos_H2BmCherry_LynTomato_CS_3dpf.lif - 1e7-1.tif" --output_path "h2b_res" --channel_id 2 --model_resolution 1 0.7 0.4 --model_path ..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\models\stardist\generic_plant_nuclei_3D --resolution_as_scale True 
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
            └── segmentation.bat        <- command to run the nuclei segmentation script
```
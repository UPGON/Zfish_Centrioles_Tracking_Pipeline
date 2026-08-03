# Nuclei manual curation (nuclei_curation.py)
Curate the nuclei detection results data.

Due to the complexity of segmentation, it is only possible to remove segmented and detected nuclei.

## Input parameters
Required

- ***centers_path*** `[pathlib.Path]`: Path to the `.csv` file containing the spot detection results. 
- ***segm_path*** `[pathlib.Path]`: Path to the TIFF image with segmented nuclei. 
- ***instructions_path*** `[pathlib.Path]`: Path to the folder containing the instructions `.txt` files.

### Instructions files
#### remove.txt
This file contains the index of the nuclei to remove with the following format:
idx 

The idx corresponds to the nucleus index in the cX_centers.csv file and annotated in the cX_segmentation.tif. 
Usually the easiest way to curate nuclei detection data is to open the cX_segmentation.tif with nucleus index annotation and take note of the indices of the nuclei to remove.

Example of remove.txt file

```
338
141
108
345
129
```
## Output results  
- ***cX_centers_curated.csv***: curated 3D (+t) coordinates and features (area & intensity) of the detected nuclei.
- ***cX_segmentation.tif***: TIFF image with segmented nuclei with centre and index annotation.

## Command usage
```bash
@echo off
python ..\..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\utils\nuclei_curation.py --centers_path CENTERS_PATH --segm_path SEGM_PATH --instructions_path INSTRUCTIONS_PATH
```

### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\src\curation\nucl_curation.py --centers_path "..\..\raw\h2b_res\c2_centers.csv" --segm_path "..\..\raw\h2b_res\c2_segmentation.tif" --instructions_path "instructions" 
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\utils\nuclei_curation.py --centers_path "..\..\raw\h2b_res\c2_centers.csv" --segm_path "..\..\raw\h2b_res\c2_segmentation.tif" --instructions_path "instructions" 
```

## Suggested folder structure

```
experiment
├── images.tif            <- original, raw images from the experiment
├── subsets           
│   └── s1_images.tif           <- subset of the original images, can notably contain its cropped version. We usually run the detection on image subset
└── results           
   └── detection
        ├── curated 
        │   └── cX_res
        │       ├── instructions
        │       │   └── remove.txt      <- instruction with the information of nuclei to remove from the semgentation
        │       └── curate_nucle.bat       <- command to curate the coordinates nuclei detection
        └── raw 
            ├── cX_res
            │   ├── cX_centers.csv      <- 3D (+t) coordinates of the detected nuclei 
            │   ├── cX_segmentation.tif        <- TIFF images with the channel image, bright field and detected nuclei annotation
            │   └── cX_params.csv       <- parameters used for the detection
            └── segmentation.bat        <- command to run the nuclei detection script
```
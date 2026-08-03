# Spot manual curation (coords_curation.py)
Curate the spot detection results data.

New detected spots can be added to the data and existing spots can be removed from the data.
The spot features (area, intensity) are recomputed from the added spots.

## Input parameters
Required

- ***centers_path*** `[pathlib.Path]`: Path to the `.csv` file containing the spot detection results. 
- ***instructions_path*** `[pathlib.Path]`: Path to the folder containing the instructions `.txt` files.
- ***vol_path*** `[pathlib.Path]`: Path to the TIFF file used for the spot detection.
- ***channel_id*** `[int]`: Channel index to process.

### Instructions files
#### add.txt
This file contains the coordinates of the spots to add with the following format:
idx [z_coordinates, y_coordinates, x_coordinates]

The idx is optional and corresponds to the coordinates of the colocalizing foci found during curation.
It is possible to write the letter coordinates (z, y, x) in front of the coordinate numbers.

Example of add.txt file

```
323[z0,y328,x398]
76[0,226,370]
83[5,219,244]
357[81,140,114]
175[z17,288,469]
```

#### remove.txt
This file contains the index of the spots to remove with the following format:
idx 

The idx corresponds to the index coordinates in the cX_centers.csv file and annotated in the cX_detection_img.tif. 
Usually the easiest way to curate spot detection data is to open the cX_detection_img.tif with spot index annotation and take note of the indices of the spots to remove.

Example of remove.txt file

```
338
141
108
345
129
```
## Output results  
- ***cX_centers_curated.csv***: curated 3D (+t) coordinates and features (area & intensity) of the detected spots.

## Command usage
```bash
@echo off
python path_to/coords_curation.py --centers_path CENTERS_PATH --instructions_path INSTRUCTIONS_PATH --vol_path VOL_PATH --channel_id CHANNEL_ID 
```

### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\src\curation\coords_curation.py --centers_path "..\..\raw\cs_res\c3_centers.csv" --instructions_path "instructions" --vol_path "..\..\..\..\subsets\s1_20260617CFHc_CetnEos_H2BmCherry_LynTomato_CS_1dpf.lif - 3e6-1.tif" --channel_id 3
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\curation\coords_curation.py --centers_path "..\..\raw\cs_res\c3_centers.csv" --instructions_path "instructions" --vol_path "..\..\..\..\subsets\s1_20260617CFHc_CetnEos_H2BmCherry_LynTomato_CS_1dpf.lif - 3e6-1.tif" --channel_id 3
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
        │       │   ├── add.txt     <- instruction with the information of spots to add to the spot detection results
        │       │   └── remove.txt      <- instruction with the information of spots to remove from the spot detection results
        │       └── curate_coords.bat       <- command to curate the coordinates spot detection
        └── raw 
            ├── cX_res
            │   ├── cX_centers.csv      <- 3D (+t) coordinates and features (area & intensity) of the detected spots 
            │   ├── cX_detection_img.tif        <- TIFF images with the channel image, bright field and detected spot annotation
            │   └── cX_params.csv       <- parameters used for the detection
            └── blob_detection_cX.bat        <- command to run the spot detection script
```
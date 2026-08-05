# Spot to nuclei pairing manual curation (cent_nuclei_pairing_curation.py)
Curate the pairing of centriolar foci to closest nuclei.

For each detected spot it is possible to re-assigned its nuclei.

## Input parameters
Required

- ***vol_path*** `[pathlib.Path]`: Path to the input TIFF file used for the nuclei detection. 
- ***segm_path*** `[pathlib.Path]`: Path to the TIFF image with segmented nuclei. 
- ***centriole_center_path*** `[pathlib.Path]`: Path to the `.csv` file containing the detected spot coordinates. Usually we want to provide the coordinates of the colocalizing spots from the `colocalization_analysis.py` script.
- ***nuclei_center_path*** `[pathlib.Path]`: Path to the `.csv` file containing the detected nuclei coordinates.
- ***instructions_path*** `[pathlib.Path]`: Path to the folder containing the instructions `.txt` files.

### Instructions files
#### pairing.txt
This file contains the curated nuclei index for a given spot (centriolar foci) index with the following format:
spot_idx new_nuclei_idx

This script allow to efficiently re-assigne nuclei to centriole. The other way around can not be archived in one line instructions and is a bit less straightforward.

Example of pairing.txt file

```
2,4
39,1
90,20
```

## Output results 
- ***images/border_annotated_pairing_detection.tif***: TIFF image with all channels and annotations indicating which foci are associated with which nucleus.
- ***plots/centrioles_per_paired_nuclei_plot.png***: histogram with the number of centrioles per nucleus with separation distance smaller than the max_pairing_dist.
- ***plots/centrioles_per_tot_nuclei_plot.png***: histogram with the number of centrioles per nucleus (no constraint on separation distance).
- ***plots/distance_plot.png***: histogram of the separation distance between spots and nuclei.
- ***plots/zoomed_distance_plot.png***: zoomed histogram of the separation distance between spots and nuclei.
- ***pairing_results.csv***: index of the paired foci-nucleus with their separation distances.
- ***statistics.csv***: general data statistics, such as the mean distance.


## Command usage
```bash
@echo off
python path_to/cent_nuclei_pairing_curation.py  --vol_path VOL_PATH --segm_path SEGM_PATH --centriole_center_path CENTRIOLE_CENTER_PATH --nuclei_center_path NUCLEI_CENTER_PATH --instructions_path INSTRUCTIONS_PATH
```

### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python C:\Users\ovola\OneDrive\Bureau\Oceane\Etudes\EPFL\Master\PDM\Thesis\Code\Zfish_Centrioles_Tracking_Pipeline\src\curation\cent_nuclei_pairing_curation.py --vol_path "..\..\..\s1_20260619CFHa_CetnEos_H2BmCherry_LynTomato_CS_3dpf.lif_-_1e7-1.tif" --segm_path "..\..\detection\h2b_res\c2_segmentation.tif" --pairing_path "..\raw\pairing_res\pairing_results.csv" --centriole_center_path "..\..\coloc\pairing_results\pairing_results.csv" --nuclei_center_path "..\..\detection\h2b_res\c2_centers.csv" --instructions_path "instructions"
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\curation\cent_nuclei_pairing_curation.py --vol_path "..\..\..\s1_20260619CFHa_CetnEos_H2BmCherry_LynTomato_CS_3dpf.lif_-_1e7-1.tif" --segm_path "..\..\detection\h2b_res\c2_segmentation.tif" --pairing_path "..\raw\pairing_res\pairing_results.csv" --centriole_center_path "..\..\coloc\pairing_results\pairing_results.csv" --nuclei_center_path "..\..\detection\h2b_res\c2_centers.csv" --instructions_path "instructions"
```

## Suggested folder structure 

```
experiment
├── images.tif            <- original, raw images from the experiment
├── subsets           
│   └── s1_images.tif           <- subset of the original images, can notably contain its cropped version. We usually run the detection on image subsets
└── results           
   └── pairing
        └── raw
            └── spot_nucl_pairing.bat        <- command to run the cent_nucl_pairing script
```
# Spot to spot pairing / Colocalization analysis (colocalization_analysis.py)
Pair detected spots from 2 different given channels and quantify the proportion and colocalization percentage of each channel's signals.

This script loads the detected spot centers from the 2 channels and pairs each spot with the closest spot in the other channel. Paired spots separated by a distance greater than a given threshold are removed. <br>
An annotated image with the colocalizing spots of the 2 channels is created and saved. <br>
Channel signal colocalization and proportion are computed and summarized in saved figures. <br>
Finally, figures with the feature comparisons between colocalizing and non-colocalizing spots are also computed and saved.

## Input parameters
Required

- ***vol_path*** `[pathlib.Path]`: Path to the input TIFF file used for the spot detection. 
- ***centers1_path*** `[pathlib.Path]`: Path to the `.csv` file containing the spot detection results from the first channel.
- ***centers2_path*** `[pathlib.Path]`: Path to the `.csv` file containing the spot detection results from the second channel.
- ***output_path*** `[pathlib.Path]`:  Path to the output folder.
- ***channel1_id*** `[int]`: First channel index to process. The index must be greater than 0 (do not use coding index formatting).
- ***channel2_id*** `[int]`: Second channel index to process. The index must be greater than 0 (do not use coding index formatting).
- ***max_pairing_distance_xy*** `[float]`: Maximum xy separation distance between spots from different channels to be considered as colocalizing.
- ***max_pairing_distance_z*** `[float]`: Maximum z separation distance between spots from different channels to be considered as colocalizing.

Optional
- ***channel1_name*** `[str]`: First channel name. This usually corresponds to the name of the fluorescent label imaged in this channel.
- ***channel2_name*** `[str]`: Second channel name. This usually corresponds to the name of the fluorescent label imaged in this channel.

## Output results 
- ***images/annotated_pairing_detection.tif***: TIFF image with the 2 channels, the bright field and circled detected foci. If annotation is on, the index of the colocalizing pair will be displayed next to the circled spots. Non-colocalizing spots will not have an index next to them.
- ***images/colocalizing_mask.tif***: TIFF image with the 2 channels, the bright field and circled colocalizing foci. 
- ***plots/area_comparison_plot.png***: boxplot of the area of colocalizing vs non-colocalizing foci for each channel.
- ***plots/colocalization_proportion_plot.png***: barplot of the colocalization percentage for each channel.
- ***plots/detection_count_plot.png***: barplot of the number of detected foci for each channel.
- ***plots/detection_proportion_plot.png***: barplot of the proportion of detected foci for each channel.
- ***plots/diff_z_distance_plot.png***: boxplot of the axial displacement between paired foci.
- ***plots/intensity_comparison_plot.png***: boxplot of the intensity of colocalizing vs non-colocalizing foci for each channel.
- ***plots/pairing_distance_xy_plot.png***: histogram of the xy separation distance between paired foci.
- ***plots/pairing_distance_z_plot.png***: histogram of the z separation distance between paired foci.
- ***plots/radius_comparison_plot.png***: boxplot of the radius of colocalizing vs non-colocalizing foci for each channel.
- ***features.csv***: mean feature measurements for each channel.
- ***pairing_results.csv***: index of the paired foci from each channel with their separation distances and mean coordinates.
- ***statistics.csv***: general data statistics, such as the proportion and colocalization percentage for each marker.

## Command usage
```bash
@echo off
python path/colocalization_analysis.py --vol_path VOL_PATH --centers1_path CENTRES1_PATH --centers2_path CENTERS2_PATH --output_path OUTPUT_PATH --channel1_id CHANNEL1_ID --channel2_id CHANNEL2_ID --max_pairing_distance_xy MAX_PAIRING_DISTANCE_XY --max_pairing_distance_z MAX_PAIRING_DISTANCE_Z --channel1_name CHANNEL1_NAME --channel2_name CHANNEL2_NAME
```


### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\src\analysis\colocalization_analysis.py --vol_path "..\..\..\subsets\s1_20260617CFHc_CetnEos_H2BmCherry_LynTomato_CS_1dpf.lif - 3e4-1.tif" --centers1_path "..\..\detection\raw\ct_res\c1_centers.csv" --centers2_path "..\..\detection\raw\cs_res\c3_centers.csv" --output_path "pairing_results" --channel1_id 1 --channel2_id 3 --max_pairing_distance_xy 0.9 --max_pairing_distance_z 2
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\analysis\colocalization_analysis.py --vol_path "..\..\..\subsets\s1_20260617CFHc_CetnEos_H2BmCherry_LynTomato_CS_1dpf.lif - 3e4-1.tif" --centers1_path "..\..\detection\raw\ct_res\c1_centers.csv" --centers2_path "..\..\detection\raw\cs_res\c3_centers.csv" --output_path "pairing_results" --channel1_id 1 --channel2_id 3 --max_pairing_distance_xy 0.9 --max_pairing_distance_z 2
```

## Suggested folder structure 

```
experiment
├── images.tif            <- original, raw images from the experiment
├── subsets           
│   └── s1_images.tif           <- subset of the original images, can notably contain its cropped version. We usually run the detection on image subsets
└── results           
   └── colocalization
        └── raw/curated     <- We usually compute the colocalization analysis on the raw and curated spot detection results to observe the differences
            └── colocalization_analysis.bat        <- command to run the spot detection script
```
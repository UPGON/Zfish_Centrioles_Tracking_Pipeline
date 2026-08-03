# Tracking (btracking.py)
Track the given centres using Bayesian motion.

## Input parameters
Required

- ***input_path*** `[pathlib.Path]`: Path to the element coordinates to track.
- ***vol_path*** `[pathlib.Path]`: Path to the input TIFF file.
- ***output_path*** `[pathlib.Path]`: Output directory.
- ***channel_id*** `[int]`: Channel index to process. The index must be greater than 0 (do not use coding index formatting).
- ***config_path*** `[pathlib.Path]`: Path to the configuration file.

Optional
- ***max_search_radius*** `[float]`: Maximum distance separating the same element in successive frames.

## Output results  
- ***data.csv***: element trajectories.
- ***track_vol.tif***: TIFF images with annotated trajectories.

## Command usage
```bash
@echo off
python path/btracking.py --input_path INPUT_PATH --vol_path VOL_PATH --output_path OUTPUT_PATH --channel_id CHANNEL_ID --output_path OUTPUT_PATH
```


### Examples
#### Absolute script path <br>
If you are connected to the upgon server with the network mapped to drive K, you can adapt this command with your username:
```bash
@echo off
python K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\src\tracking\btracking.py --input_path "..\colocalization\curated\curated_pairing_results\pairing_results.csv" --vol_path "..\..\..\subsets\20260617CFHb_CetnEos_H2BmCherry_LynTomato_CS_1dpf_TL.lif - 2e1-1.tif" --output_path "tracking_res" --channel_id 3 --config_path K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\models\btrack\cell_config.json
```

#### Relative script path <br>
Adapt the number of `..\` to your own folder location:
```bash
@echo off
python ..\..\..\..\..\..\..\..\code\Zfish_Centrioles_Tracking_Pipeline\src\tracking\btracking.py --input_path "..\colocalization\curated\curated_pairing_results\pairing_results.csv" --vol_path "..\..\..\subsets\20260617CFHb_CetnEos_H2BmCherry_LynTomato_CS_1dpf_TL.lif - 2e1-1.tif" --output_path "tracking_res" --channel_id 3 --config_path K:\users\voland\code\Zfish_Centrioles_Tracking_Pipeline\models\btrack\cell_config.json
```


## Suggested folder structure 

```
experiment
├── images.tif            <- original, raw images from the experiment
├── subsets           
│   └── s1_images.tif           <- subset of the original images, can notably contain its cropped version. We usually run the detection on image subsets
└── results           
   └── tracking
        └── raw 
            └── btracking.bat        <- command to run the tracking script
```
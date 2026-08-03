
## 7. Project Organization
This document summarizes the project folder and file organisation. 

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── docs               <- The project documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks  
│   │
│   ├── colocalization_analysis
│   │   ├── 4-6hpf_Cetn2GFP_CS.ipynb                   <- main notebook for the exploration and implementation of the colocalization analysis. Contains spot detector exploration and first figure drafts
│   │   ├── area_remover_example.ipynb                   <- explore different thresholding methods to compute spot area and remove them when requiered
│   │   └── features_measurements.ipynb                   <- explore different thesholding methods to compute spot area and intensity and design some figures
│   │
│   ├── detection
│   │   ├── 20260506CFHc_CetnEos_H2BmCherry_CS_30hpf_25-63x.ipynb                   <- explore different methods to detect and segment nuclei
│   │   └── merging_centers.ipynb                  <- explore different algorithm to merge points toghether
│   ├── microscope_parameters
│   │   └── microscope_parameters.ipynb                  <- explore the effect of the microscope parameters on the data
│   ├── omero_connection
│   │   └── omero_connection.ipynb                  <- connect to your omero session to open data
│   ├── pairing
│   │   └── pairing_centrioles_nuclei.ipynb                  <- explore methods to pair centriolar foci to the closest nuclei 
│   ├── pairing
│   │   ├── 4-6hpf_Cetn2GFP_CS.ipynb                  <- explore methods to track centriolar foci over frames
│   │   └── 20260506CFHc_CetnEos_H2BmCherry_CS_30hpf_25-63x.ipynb                  <- explore methods to track nuclei over frames    
│   └── visualization
│       ├── global_plots.ipynb                  <- create figures illustrating global data behavior. Mostly used to generate the figures for the thesis report
│       └── napari.ipynb                  <- generate 3D movies of the data using napari
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes zf_centrioles_tracking_pipeline a Python module
    │
    ├── analysis
    │   └── colocalisation_analysis.py                   <- Pair 2 foci groups toghether and quantify their colocalisation
    │
    ├── curation                
    │   ├── __init__.py 
    │   ├── coords_curation.py          <- Curated csv file containing coordinates
    │   ├── coords_reindexing.py            <- Reindex csv file containing coordinates      
    │   ├── nuclei_curation.py            <- Curated measured data of nuclei (coordinate & segmantion)         
    │   └── pairing_reindexing.py            <-  Reindex csv file containing paired points          
    │
    ├── dataset   
    │   └── omero_connection
    │        └── omero_connection.py <- connect to an omero session and open data
    │
    ├── detection                
    │   ├── __init__.py 
    │   ├── blob_detection.py          <- Detect foci          
    │   └── stardist_segmentation.py            <- Detect and segmente nuclei
    │
    ├── pairing                
    │   ├── __init__.py 
    │   ├── cent_nucl_pairing.py          <- Pair foci to their closest nuclei          
    │   └── points_pairing.py            <- Pair points toghther
    │
    ├── pre_processing                
    │   ├── __init__.py 
    │   ├── cropping.py          <- Crop images
    │   ├── denoising.py            <- Denoise images using Gaussian or Median filter          
    │   ├── interpolation.py            <- Interpolate images           
    │   ├── stacking.py            <- Stack images in a folder into a single images          
    │   └── thresholding.py             <- Threshold images
    │
    ├── tracking                
    │   ├── __init__.py 
    │   └── btracking.py          <- Track spots using Bayesian tracker
    │
    ├── utils                    
    │   └── utils.py             <- Global utilitary function
    │
    ├── visualization                
    │   ├── __init__.py 
    │   ├── plots.py          <- Utilitary function to create plots        
    │   └── visualization.py             <- Utilitary functions to annotated images 
    │
    └── constants.py                <- Constants of the scripts
```

--------
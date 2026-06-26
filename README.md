# ZFISH_CENTRIOLES_TRACKING_PIPELINE

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project allow the analysis and quantitfication of centrioles/centriosomes foci in zebrafish images, specially in the muscles. These scripts can handle 2D, 3D images and 3D timelapse when relevant (e.g cropping, tracking).
The main tasks offered by the pipeline are:
- detection of foci (bright spots on dark backgorund)
- detection of nuclei
- mapping of foci to its closest nuclei
- tracking of foci or nuclei
Additionally mutliple utilitary functions are also provided to allow processing, visualisation or quantification of the data.

Please consult the README associated with the task you wish to perfom to know the usage.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         zf_centrioles_tracking_pipeline and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes zf_centrioles_tracking_pipeline a Python module
    │
    ├── analysis
    │   ├── colocalisation_analysis.py                   <- Pair 2 foci groups toghether and quantify their colocalisation
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
    │   ├── __init__.py 
    │   ├── coords_curation.py          <- Curated csv file containing coordinates
    │   ├── coords_reindexing.py            <- Reindex csv file containing coordinates      
    │   ├── nuclei_curation.py            <- Curated measured data of nuclei (coordinate & segmantion)         
    │   ├── pairing_reindexing.py            <-  Reindex csv file containing paired points          
    │   └── utils.py             <- Global utilitary function
    │
    ├── visualization                
    │   ├── __init__.py 
    │   ├── plots.py          <- Utilitary function to create plots        
    │   └── visualization.py             <- Utilitary functions to annotated images 
    │
    ├── config.py               <- Store useful variables and configuration
    │
    └── constants.py                <- Constants of the scripts
```

--------


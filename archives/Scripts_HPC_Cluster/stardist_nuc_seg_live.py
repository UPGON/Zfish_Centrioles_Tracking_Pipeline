#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:23:46 2024

@author: floriancurvaia
"""


import numpy as np
from tifffile import imread, imwrite
import sys
from stardist.models import StarDist3D
from csbdeep.utils import normalize
import argparse

from pathlib import Path

CLI=argparse.ArgumentParser()
CLI.add_argument('idx', type=int)
CLI.add_argument(
  "fld",  # name on the CLI - drop the `--` for positional/required parameters
  #nargs=1,  # 0 or more values expected => creates a list
  type=str,  # default if nothing is provided
)

args = CLI.parse_args()

tp=str(args.idx).zfill(5)
path_in=Path(args.fld)
path_in = Path(r"\\sv-nas1.rcp.epfl.ch\upgon_archives\common\Curvaia_Florian\Images\Live\20240321_Transplants\20240321_172724_Transplants_TgCentrinEos_H2BmCherry\e2-1_FLUO\e2-1_time_registration\3D\Muscles_part2\volumes")
fn=path_in / ("ome-tiff.companion-track-crop-crop--C01--T"+tp+".tif")

path_in=fn.parent

im_name=fn.name.split(".tif")[0]

scale=(0.75, 0.173, 0.173)
x=imread(fn)

#img_H2B_med = rank.median(img_as_uint(img_H2B), cube(3)) # disk(2)
#fp=ndimage.generate_binary_structure(3, 1)
#img_H2B_med = ndimage.median_filter(img_H2B, footprint=fp)
#img_H2B_med = median(img_H2B, cube(3))
#x = normalize(img_H2B)
x = normalize(x)


if __name__ == "__main__":
    # original name 
    # unet_depth_3_classes_True_augment_2_grid_2_2_2_patch_112_epochs_200
    # The file was renamed to allow its addition in GitHub under the archives folder
    fpath="../stardist_models/unet_model/"
    #fpath="models/early_embryo_model/"
    #fpath="models/late_blastocyst_model/"
    model = StarDist3D(None, fpath)
        
    n_tiles = tuple(int(np.ceil(s/256)) for s in x.shape)
        
    print(f'predicting {x.shape} with {n_tiles} tiles')
    #y, _ = model.predict_instances(x, scale=(1,.3,.3), n_tiles=n_tiles) #SP8
    
    y, _ = model.predict_instances(x, scale=(1,0.23,0.23), n_tiles=n_tiles) #SP8 21.02.2024  scale=(1,0.2729824561403509,0.2729824561403509)
    #y2, _ = model.predict_instances(x, scale=(1,.23,.23), n_tiles=n_tiles) #SPIM
  
res_path = Path(r"K:\users\voland\images\florians_data\Transplants_e1_2\C01_T00001")
np.save( res_path / "npy_seg" / ("T"+str(tp).zfill(5)+"_nuc_seg.npy"), y.astype("int16"))

imwrite( res_path / "tif_seg" / ("T"+str(tp).zfill(5)+"_nuc_seg.tif"), y.astype("int16"))








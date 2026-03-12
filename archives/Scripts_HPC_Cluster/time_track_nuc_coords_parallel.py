#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:49:10 2024

@author: floriancurvaia
"""

import btrack

from pathlib import Path
from tifffile import imread
import numpy as np
import dask.array as da
from tifffile import imwrite 

import argparse

import pandas as pd

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

fn=path_in / ("T"+tp+"_nuc_seg.npy")

path_out=Path(path_in.parent/"Nuc_seg_time_track/")


nuc_seg_tp=np.load(fn)
data=np.load(path_in.parent/"nuc_coords_Muscles_V2.npy")

scale=(0.75, 0.173, 0.173)

all_coords=pd.DataFrame(data, columns=["ID", "T", "Z", "Y", "X"])

filt_coords=all_coords.loc[all_coords["T"]==int(tp)]
#copy_C3=stack_C3.copy()

dest_array=np.zeros_like(nuc_seg_tp, dtype="uint16")#.compute()
Z_lim=nuc_seg_tp.shape[0]
for nuc_coords in filt_coords.to_numpy():
    ID, T, Z, Y, X = nuc_coords
    if int(Z/scale[0])>Z_lim-1:
        print(nuc_coords)
        continue
    time_frame_ID= nuc_seg_tp[int(np.round(Z/scale[0])), int(np.round(Y/scale[1])), int(np.round(X/scale[2]))]
    if time_frame_ID==0:
        continue
    dest_array[nuc_seg_tp==time_frame_ID]=int(ID) #.compute()
    
    
    
np.save( path_out / ("T"+tp+"_nuc_seg.npy"), dest_array)

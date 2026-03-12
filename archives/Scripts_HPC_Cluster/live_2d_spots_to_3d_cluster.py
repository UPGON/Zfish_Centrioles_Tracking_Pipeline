#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:03:09 2024

@author: floriancurvaia
"""


from tifffile import imread
import numpy as np
import dask.array as da
import argparse
import pandas as pd

from pathlib import Path

CLI=argparse.ArgumentParser()
CLI.add_argument('idx', type=int)
CLI.add_argument(
  "fld",  # name on the CLI - drop the `--` for positional/required parameters
  #nargs=1,  # 0 or more values expected => creates a list
  type=str,  # default if nothing is provided
)
CLI.add_argument("spfn", type=str)
args = CLI.parse_args()

tp=int(args.idx)
path_in=Path(args.fld)
spots_filename=args.spfn

fn=path_in / ("ome-tiff.companion-track-crop--C00--T"+str(tp).zfill(5)+".tif")

path_in=fn.parent
path_in_spots = Path(path_in.parent/spots_filename)
all_spots=pd.read_csv(path_in_spots)
spots_tp=all_spots.loc[all_spots["POSITION_T"]==tp]
N_time_points=1
all_time_points=list(range(1, N_time_points+1))
spots_time_points=[]

    
cetn=imread(fn)


scale=(0.75, 0.173, 0.173)



spots_tp[["X", "Y"]] = (spots_tp[["POSITION_X", "POSITION_Y"]] /scale[-1]).astype(int)
spots_xy=spots_tp[["X", "Y"]].to_numpy()
#spots_xy/=0.155
cetn_zmax=cetn.argmax(0)
spots_tp["Z"]=cetn_zmax[spots_xy[:,::-1].astype(int)[:,0], spots_xy[:,::-1].astype(int)[:,1]]
spots_zyx=spots_tp[["Z", "Y", "X"]].to_numpy()
spots_tp["POSITION_Z"]=spots_tp["Z"]*scale[0]

spots_tp.to_csv(path_in.parent / ("Centrioles_spots_3D") / ("3D_spots_T"+str(tp).zfill(5)+".csv"))

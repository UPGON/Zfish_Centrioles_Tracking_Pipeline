#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:24:24 2024

@author: floriancurvaia
"""


import btrack
import matplotlib.cm as cm
import napari
from pathlib import Path
from tifffile import imread
from napari_animation import Animation
import numpy as np
import dask.array as da
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy import stats
import pickle 
from scipy import spatial
#import plotly.express as px

plt.ioff()
#stack = imread("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/e2-1_cells_of_interest_time_registered_3D.tif")

Cell_ID=540

path_in_C3=Path("//Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/Nuc_seg_time_track")
path_in_C2=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/volumes_V2")
path_in_C1=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/volumes_V2")
im_prefix=""


path_out_im="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/"


scale=(0.75, 0.173, 0.173)
new_scale=(1, 0.75, 0.173, 0.173)

spots_track_coords_737=pd.read_csv(path_out_im + "all_cur_spots_id"+str(Cell_ID)+".csv")[["ID", "T", "Z", "Y", "X"]]
spots_tzyx_nuc_737=spots_track_coords_737.to_numpy()[:, 1:]/new_scale
spots_track_coords_737["Nuc_ID"]=0
spots_track_coords_737["Dist_nuc"]=np.nan
spots_nuc_npy_coords_737=spots_track_coords_737[["ID","T","Z", "Y", "X"]].astype(int)




#nuc_seg_file="/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/Nuc_seg_time_track.tif"
#path_in_C3=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/tif_seg/")
#path_in_C2=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/")
#path_in_C1=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/")

#path_out=Path("/scratch/curvaia/Transplants_e1_2/")

filenames_C3 = sorted(path_in_C3.glob('*.npy'))

filenames_C2 = sorted(path_in_C2.glob('*C01*'))

filenames_C1 = sorted(path_in_C1.glob('*C00*'))


def read_one_image_C2(block_id, filenames=filenames_C2, axis=0):
    # a function that reads in one chunk of data
    path = filenames[block_id[axis]]
    image = imread(path)
    return np.expand_dims(image, axis=axis) #np.transpose(np.expand_dims(image, axis=axis), (2,1,0))

def read_one_image_C1(block_id, filenames=filenames_C1, axis=0):
    # a function that reads in one chunk of data
    path = filenames[block_id[axis]]
    image = imread(path)
    return np.expand_dims(image, axis=axis)


def read_one_image_C3(block_id, filenames=filenames_C3, axis=0):
    # a function that reads in one chunk of data
    path = filenames[block_id[axis]]
    image = np.load(path)
    return np.expand_dims(image, axis=axis)



sample_C2 = imread(filenames_C2[0]) #np.transpose(imread(filenames_C2[0]), (2,1,0))

sample_C1 = imread(filenames_C1[0])

sample_C3 = np.load(filenames_C3[0])

stack_C2 = da.map_blocks(
    read_one_image_C2,
    dtype=sample_C2.dtype,
    chunks=((1,) * len(filenames_C2), *sample_C2.shape)
)
stack_C2=stack_C2.astype("int16")

stack_C1 = da.map_blocks(
    read_one_image_C1,
    dtype=sample_C1.dtype,
    chunks=((1,) * len(filenames_C1), *sample_C1.shape)
)


stack_C3 = da.map_blocks(
    read_one_image_C3,
    dtype=sample_C3.dtype,
    chunks=( (1,) * len(filenames_C3), *sample_C3.shape )
)


stack_C1=stack_C1.astype("int16")


### Distance to nucleus part
columns_to_fill=["Nuc_ID", "Dist_nuc"]
idx_cols=[]
for col in columns_to_fill:
    idx_cols.append(spots_track_coords_737.columns.get_loc(col))

with open(path_out_im+'All_nuc_IDs_tp_id'+str(Cell_ID)+'.pkl', 'rb') as f:
    dict_nuc_per_tp = pickle.load(f)
"""
dict_spot_to_nuc_737={}
with open(path_out_im + "Closest_nuc_IDs_per_spot_per_tp_id540_Curated.txt" ) as f:
    lines=f.readlines()
    tp=None
    for line in lines:
        row=line.strip()
        chunks=row.split(" ")
        line_start=chunks[0]
        if not line_start:
            continue
        if line_start == "T:":
            tp=int(chunks[1])
            dict_spot_to_nuc_737[tp]={}
            continue
        
        spot_id=int(line_start.split(":")[0])
        nuc_id=int(chunks[1])
        dict_spot_to_nuc_737[tp][spot_id]=nuc_id

with open(path_out_im+'dict_spots_per_nuc_per_tp_id540.pkl', 'wb+') as f:
    pickle.dump(dict_spot_to_nuc_737, f) 
"""    

for i in range(350):
    spots_737_tp=spots_track_coords_737.loc[spots_track_coords_737["T"]==i]
    nuc_seg_tp=stack_C3[i].compute()
    
    nuc_id_tp=dict_nuc_per_tp[i]
    if len(nuc_id_tp)==0:
        for ind, p in spots_737_tp.iterrows():
            spots_track_coords_737.iloc[ind, idx_cols]=[np.nan, np.nan]
        continue
    
    #all_nucs_index=np.argwhere(nuc_seg_tp)
    all_nucs_index=np.argwhere(np.isin(nuc_seg_tp, nuc_id_tp))
    all_nucs_lab=nuc_seg_tp[tuple(all_nucs_index.T)]#.compute()
    all_nucs_coords=all_nucs_index * scale
    tree = spatial.KDTree(all_nucs_coords)
    
    for ind, p in spots_737_tp.iterrows():
        dist_p, loc_ind= tree.query(p[["Z", "Y", "X"]].to_numpy())
        #spots_track_coords_135.iloc[ind][["Nuc_ID", "Dist_nuc"]]=[loc_ind, dist_p]
        nuc_lab=all_nucs_lab[loc_ind]
        spots_track_coords_737.iloc[ind, idx_cols]=[nuc_lab, dist_p]
    #tree.query([(21,21)])

#test=np.zeros_like(stack_C1[0]).astype(int)
"""

spots_track_coords_737["Fluo"]=np.nan
spots_track_coords_737["Fluo_mean"]=np.nan

spots_track_coords_135["Fluo"]=np.nan
spots_track_coords_135["Fluo_mean"]=np.nan

columns_to_fill=["Fluo", "Fluo_mean"]
idx_cols=[]
for col in columns_to_fill:
    idx_cols.append(spots_track_coords_135.columns.get_loc(col))
    
    

"""

### Intensity Value part
dict_spots_values_per_tp_737={}

for i in range(350):
    dict_spots_values_per_tp_737[i]={}
    
d, h, w = stack_C1[0].shape #100, 100, 10
depthnums, colnums, rownums = np.meshgrid(range(d), range(h), range(w), indexing='ij')
r=2.5 #Pseudo radius of the spot

for i in range(350):
    spots_737_tp=spots_track_coords_737.loc[spots_track_coords_737["T"]==i]
    centrin=stack_C1[i].compute()
    zmax, ymax, xmax=centrin.shape
    
    for ind, p in spots_737_tp.iterrows():
        cz, cy, cx= (p[["Z", "Y", "X"]].to_numpy() /scale).flatten().astype(int)
        dist = np.sqrt(
                ((colnums.flatten() - cy)*0.75) ** 2 + 
                ((rownums.flatten() - cx)*0.75) ** 2 +
                ((depthnums.flatten() - cz)*1.66) ** 2 
                )
        #spots_track_coords_135.iloc[ind][["Nuc_ID", "Dist_nuc"]]=[loc_ind, dist_p]
        keep_mask = (dist < r).reshape((d, h, w))
        indices = np.where(keep_mask)
        if not zmax>max(indices[0]):
            new_ind=[]
            for j in range(len(indices)):
                new_ind.append(indices[j][indices[0]<zmax])
            indices=tuple(new_ind)
        if not ymax>max(indices[1]):
            new_ind=[]
            for j in range(len(indices)):
                new_ind.append(indices[j][indices[1]<ymax])
            indices=tuple(new_ind)
        if not xmax>max(indices[2]):
            new_ind=[]
            for j in range(len(indices)):
                new_ind.append(indices[j][indices[2]<xmax])
            indices=tuple(new_ind)
        values=centrin[indices]
        s_ID=p.ID
        dict_spots_values_per_tp_737[i][s_ID]=values
        
        """
        q=np.quantile(values, 0.875)
        val_indices=np.where((centrin*keep_mask)>q)
        new_vals=centrin[val_indices]
        sum_fluo=np.sum(new_vals)
        mean_fluo=np.mean(new_vals)
        spots_track_coords_737.iloc[ind, idx_cols]=[sum_fluo, mean_fluo]
        """
    del centrin
        
#del dict_spots_values_per_tp_737[103][13]

with open(path_out_im+'dict_spots_fluo_values_r2_5_xy0_75_z1_66_per_tp_id'+str(Cell_ID)+'.pkl', 'wb+') as f:
    pickle.dump(dict_spots_values_per_tp_737, f) 
    
#spots_track_coords_737.drop(spots_track_coords_737.loc[spots_track_coords_737.ID==13].index, axis=0, inplace=True)

spots_track_coords_737.to_csv(path_out_im + "all_cur_spots_id"+str(Cell_ID)+"_w_dist.csv")















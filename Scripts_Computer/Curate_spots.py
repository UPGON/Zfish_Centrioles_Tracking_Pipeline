#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:42:52 2024

@author: floriancurvaia
"""


###PART 1 of the script
import btrack

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
#import mpl_scatter_density # adds projection='scatter_density'
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy import stats

plt.ioff()
#stack = imread("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/e2-1_cells_of_interest_time_registered_3D.tif")

Muscle_cells=True

N_tp=350

path_in_spots=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/Centrioles_spots_3D")
path_in_C3=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/Tif/nuc_seg_npy/")
path_in_C2=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/Tif/volumes_V2/")
path_in_C1=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/Tif/volumes_V2/")
with btrack.io.HDF5FileHandler(
  '/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_cells_v2-1.h5', 'r', obj_type='obj_type_1'
) as reader:
  tracks = reader.tracks
  objs = reader.objects
im_prefix=""
    

path_out_im="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/"
path_config=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/")

all_spots=[]
for i in range(N_tp):
    tp_spots=pd.read_csv(path_in_spots / ("3D_spots_T"+str(i).zfill(5)+".csv"))
    all_spots.append(tp_spots)

#If you have already started to curate the spots of a particular cell, replace the Cell_ID below by the one of the cell you are studying,
# uncomment the three following lines below by removing the hashtag (#), and after running the part 1, directly run the last chunk of part 2 to re-start curation where you left it.
#Cell_ID=737 
#spots_tzyx_nuc_df=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/cur_spots_t349_id"+str(Cell_ID)+".csv")
#spots_tzyx_nuc=spots_tzyx_nuc_df[["T","Z", "Y", "X"]].to_numpy()

all_spots=pd.concat(all_spots, axis=0, ignore_index=True)
#all_spots=all_spots.loc[all_spots.POSITION_X>22]
all_spots[["POSITION_T", "Z", "Y", "X"]]=all_spots[["POSITION_T", "Z", "Y", "X"]].astype("float64")

scale=(0.75, 0.173, 0.173)



spots_tzyx=all_spots[["POSITION_T","Z", "Y", "X"]].to_numpy()


#path_out=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/")


#nuc_seg_file="/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/Nuc_seg_time_track.tif"
#path_in_C3=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/tif_seg/")
#path_in_C2=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/")
#path_in_C1=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/")

#path_out=Path("/scratch/curvaia/Transplants_e1_2/")

filenames_C3 = sorted(path_in_C3.glob('*.npy'))

filenames_C2 = sorted(path_in_C2.glob('*C01*'))

filenames_C1 = sorted(path_in_C1.glob('*C00*'))

def read_one_image_C3(block_id, filenames=filenames_C3, axis=0):
    # a function that reads in one chunk of data
    path = filenames[block_id[axis]]
    image = np.load(path)
    return np.expand_dims(image, axis=axis)

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



# load the first image (assume rest are same shape/dtype)
sample_C3 = np.load(filenames_C3[0]) 

sample_C2 = imread(filenames_C2[0]) #np.transpose(imread(filenames_C2[0]), (2,1,0))

sample_C1 = imread(filenames_C1[0])

stack_C3 = da.map_blocks(
    read_one_image_C3,
    dtype=sample_C3.dtype,
    chunks=( (1,) * len(filenames_C3), *sample_C3.shape )
)
stack_C3=stack_C3.astype("int16")

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
stack_C1=stack_C1.astype("int16")

scale=(0.75, 0.173, 0.173)

#objs=btrack.io.segmentation_to_objects(segmentation=stack_C3, intensity_image=stack_C2, scale=scale)

xlim=stack_C1.shape[-1]
ylim=stack_C1.shape[-2]
zlim=stack_C1.shape[1]

with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  tracker.configure(path_config /'cell_config.json')

  # append the objects to be tracked
  tracker.append(objs)

  # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
  tracker.volume = ((0, xlim), (0, ylim), (0, zlim))

  # track them (in interactive mode)
  tracker.track_interactive(step_size=10)

  # generate hypotheses and run the global optimizer
  tracker.optimize()

  # store the data in an HDF5 file
  #tracker.export( path_in_spots / 'tracks.h5', obj_type='obj_type_1')

  # get the tracks as a python list
  tracks = tracker.tracks

  # optional: get the data in a format for napari
  data, properties, graph = tracker.to_napari()
  
  #tracker.export('/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_non_muscle_cells.h5', obj_type='obj_type_1')
  

nuc_coords=pd.DataFrame(data, columns=["ID", "T", "Z", "Y", "X"])
#np.save('/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/nuc_coords_nm_cell.npy', data)


viewer = napari.Viewer(ndisplay=2)


image_layer = viewer.add_image(stack_C1, scale=scale, colormap='green', blending='additive',visible=True, contrast_limits=[90, 400])

image_layer = viewer.add_image(stack_C2, scale=scale, colormap='magenta', blending='additive',visible=True, contrast_limits=[90, 500])

image_layer = viewer.add_labels(stack_C3, scale=scale, blending='additive',visible=True)
viewer.add_tracks(data, properties=properties, graph=graph, visible=True, colormap="turbo", blending="additive") # scale=scale


###PART 2 of the script

N_spots_tp=[]
N_nucs_tp=[]
for t in range(N_tp):
    N_spots_tp.append(len(all_spots.loc[all_spots.POSITION_T==t]))
    N_nucs_tp.append(len(nuc_coords.loc[nuc_coords["T"]==t]))
    


N_spots_per_nuc_T={}
for a, b in data[:,:2]:
    if b not in N_spots_per_nuc_T.keys():
        N_spots_per_nuc_T[b]={}
    N_spots_per_nuc_T[b][a]=0

dist_spots_per_nuc_T={}
for a, b in data[:,:2]:
    if b not in dist_spots_per_nuc_T.keys():
        dist_spots_per_nuc_T[b]={}
    dist_spots_per_nuc_T[b][a]=[]

for i in range(N_tp):
    spots_tp=all_spots.loc[all_spots["POSITION_T"]==i]
    nuc_tp=nuc_coords.loc[nuc_coords["T"]==i]
    dist_0=distance.cdist(nuc_tp[["Z", "Y", "X"]].to_numpy(), spots_tp[["POSITION_Z", "POSITION_Y", "POSITION_X"]].to_numpy())
    nuc_min_id=np.argmin(dist_0, 0)
    nuc_ID=nuc_tp.iloc[nuc_min_id].ID.to_numpy()
    for j, k in zip(nuc_ID, range(len(nuc_ID))):
        N_spots_per_nuc_T[i][j]+=1
        dist_spots_per_nuc_T[i][j].append(dist_0[int(nuc_min_id[k]), k])
        

mean_spots_per_nuc_T=[np.mean(list(N_spots_per_nuc_T[i].values())) for i in range(N_tp)]
mean_dist_spots_nuc_T=[np.mean([x for xs in list(dist_spots_per_nuc_T[i].values()) for x in xs]) for i in range(N_tp)]
all_dist_spots_nuc_T=[x for i in range(N_tp) for xs in list(dist_spots_per_nuc_T[i].values()) for x in xs ]
tp_all_spots=[i for i in range(N_tp) for xs in list(dist_spots_per_nuc_T[i].values()) for x in xs]
all_N_spots_per_nuc_T=[x for i in range(N_tp) for x in list(N_spots_per_nuc_T[i].values()) ]

tp_all_nuc=[i for i in range(N_tp) for xs in list(dist_spots_per_nuc_T[i].values())]


nuc_id=737
#ids_to_track=(27, 4)
#ids_to_track=[10]
#ids_to_track=(135, 5)

dict_ids_to_track_737 = {
    i: (
        (737, 312, 96, 415, 449, 336, 7) if i < 81 or 87 < i < 92 else
        (737, 415, 449, 336, 312, 96, 7) if i < 113 else
        (737, 449, 415, 336, 312, 96, 7)
    )
    for i in range(N_tp)
}

dict_ids_to_track_737 = {
    i: tuple([10])
    for i in range(N_tp)
}

dict_ids_to_track_737 = {
    i: (
        tuple([583]) if i > 265  else
        tuple([119]) if i > 38 else
        tuple([59]) if i > 29 else
        tuple([14]) if i > 1  else
        tuple([5])
        
    )
    for i in range(N_tp)
}

mean_spots_per_nuc=[]
mean_spots_dist_nuc=[]
all_spots_dist_nuc=[]
tp_all_spots_dist=[]


for i in range(N_tp):
    dic_N_tp=N_spots_per_nuc_T[i]
    dic_D_tp=dist_spots_per_nuc_T[i]
    repo=-1
    ids_to_track=dict_ids_to_track_737[i]
    for j in ids_to_track:
        if j in  dic_N_tp.keys():
            repo=dic_N_tp[j]
            mean_spots_per_nuc.append(repo)
            mean_spots_dist_nuc.append(np.mean(dic_D_tp[j]))
            for k in range(len(dic_D_tp[j])):
                all_spots_dist_nuc.append(dic_D_tp[j][k])
                tp_all_spots_dist.append(i)
            break
    if repo==-1:
        mean_spots_per_nuc.append(np.nan)
        mean_spots_dist_nuc.append(np.nan)





spots_close_nuc=[] 


N_spots_per_single_nuc_T=np.zeros((N_tp))

dist_spots_per_single_nuc_T=[]
tp_dist_spots_per_single_nuc=[]
"""
for i in range(N_tp):
    #spots_tp=spots_tzyx_nuc_df.loc[spots_tzyx_nuc_df["T"]==i]
    spots_tp=all_spots.loc[all_spots["POSITION_T"]==i]
    nuc_tp=nuc_coords.loc[nuc_coords["T"]==i]
    dist_0=distance.cdist(nuc_tp[["Z", "Y", "X"]].to_numpy(), (spots_tp[["Z", "Y", "X"]]*scale).to_numpy())
    nuc_min_id=np.argmin(dist_0, 0)
    all_tp_nuc_ID=nuc_tp.iloc[nuc_min_id].ID.to_numpy()
    ids_to_track=dict_ids_to_track_737[i]
    for j in ids_to_track:
        if j in nuc_tp.ID.to_numpy():
            nuc_index=nuc_tp.reset_index().index[nuc_tp.ID==j].values[0]
            spots_index=np.where(all_tp_nuc_ID==j)[0]
            for k in spots_index:
                if dist_0[nuc_index, k]<20:
                    spots_close_nuc.append(spots_tp.iloc[[k]])
                    N_spots_per_single_nuc_T[i]+=1
                    dist_spots_per_single_nuc_T.append(dist_0[nuc_index, k])
                    tp_dist_spots_per_single_nuc.append(i)
            coords=nuc_tp.loc[nuc_tp.ID==j]
            break
        
    fake_tp=i
    while coords.empty:
        fake_tp-=1
        #spots_tp=spots_tzyx_nuc_df.loc[spots_tzyx_nuc_df["T"]==i]
        spots_tp=all_spots.loc[all_spots["POSITION_T"]==i]
        nuc_tp=nuc_coords.loc[nuc_coords["T"]==fake_tp]
        dist_0=distance.cdist(nuc_tp[["Z", "Y", "X"]].to_numpy(), (spots_tp[["Z", "Y", "X"]]*scale).to_numpy())
        nuc_min_id=np.argmin(dist_0, 0)
        all_tp_nuc_ID=nuc_tp.iloc[nuc_min_id].ID.to_numpy()
        ids_to_track=dict_ids_to_track_737[fake_tp]
        for j in ids_to_track:
            if j in nuc_tp.ID.to_numpy():
                nuc_index=nuc_tp.reset_index().index[nuc_tp.ID==j].values[0]
                spots_index=np.where(all_tp_nuc_ID==j)[0]
                for k in spots_index:
                    if dist_0[nuc_index, k]<20:
                        spots_close_nuc.append(spots_tp.iloc[[k]])
                        N_spots_per_single_nuc_T[i]+=1
                        dist_spots_per_single_nuc_T.append(dist_0[nuc_index, k])
                        tp_dist_spots_per_single_nuc.append(i)
                coords=nuc_tp.loc[nuc_tp.ID==j]
                break
            
            
                    
            #dist_0=distance.cdist(nuc_tp.loc[nuc_tp.ID==j][["Z", "Y", "X"]].to_numpy(), spots_tp[["POSITION_Z", "POSITION_Y", "POSITION_X"]].to_numpy())
            #spots_ids=np.where(dist_0.flatten()<20)
"""
for i in range(N_tp):
    #spots_tp=spots_tzyx_nuc_df.loc[spots_tzyx_nuc_df["T"]==i]
    spots_tp=all_spots.loc[all_spots["POSITION_T"]==i]
    nuc_tp=nuc_coords.loc[nuc_coords["T"]==i]
    dist_0=distance.cdist(nuc_tp[["Z", "Y", "X"]].to_numpy(), (spots_tp[["Z", "Y", "X"]]*scale).to_numpy())
    nuc_min_id=np.argmin(dist_0, 0)
    all_tp_nuc_ID=nuc_tp.iloc[nuc_min_id].ID.to_numpy()
    ids_to_track=dict_ids_to_track_737[i]
    for j in ids_to_track:
        if j in nuc_tp.ID.to_numpy():
            nuc_index=nuc_tp.reset_index().index[nuc_tp.ID==j].values[0]
            spots_index=range(dist_0.shape[1])#np.where(all_tp_nuc_ID==j)[0]
            for k in spots_index:
                if dist_0[nuc_index, k]<10:
                    spots_close_nuc.append(spots_tp.iloc[[k]])
                    N_spots_per_single_nuc_T[i]+=1
                    dist_spots_per_single_nuc_T.append(dist_0[nuc_index, k])
                    tp_dist_spots_per_single_nuc.append(i)
            coords=nuc_tp.loc[nuc_tp.ID==j]
            break
        
    fake_tp=i
    while coords.empty:
        fake_tp-=1
        #spots_tp=spots_tzyx_nuc_df.loc[spots_tzyx_nuc_df["T"]==i]
        spots_tp=all_spots.loc[all_spots["POSITION_T"]==i]
        nuc_tp=nuc_coords.loc[nuc_coords["T"]==fake_tp]
        dist_0=distance.cdist(nuc_tp[["Z", "Y", "X"]].to_numpy(), (spots_tp[["Z", "Y", "X"]]*scale).to_numpy())
        nuc_min_id=np.argmin(dist_0, 0)
        all_tp_nuc_ID=nuc_tp.iloc[nuc_min_id].ID.to_numpy()
        ids_to_track=dict_ids_to_track_737[fake_tp]
        for j in ids_to_track:
            if j in nuc_tp.ID.to_numpy():
                nuc_index=nuc_tp.reset_index().index[nuc_tp.ID==j].values[0]
                spots_index=range(dist_0.shape[1])#np.where(all_tp_nuc_ID==j)[0]
                for k in spots_index:
                    if dist_0[nuc_index, k]<10:
                        spots_close_nuc.append(spots_tp.iloc[[k]])
                        N_spots_per_single_nuc_T[i]+=1
                        dist_spots_per_single_nuc_T.append(dist_0[nuc_index, k])
                        tp_dist_spots_per_single_nuc.append(i)
                coords=nuc_tp.loc[nuc_tp.ID==j]
                break
            
            
                    
            #dist_0=distance.cdist(nuc_tp.loc[nuc_tp.ID==j][["Z", "Y", "X"]].to_numpy(), spots_tp[["POSITION_Z", "POSITION_Y", "POSITION_X"]].to_numpy())
            #spots_ids=np.where(dist_0.flatten()<20)
            
spots_close_nuc=pd.concat(spots_close_nuc, ignore_index=True)

spots_tzyx_nuc=spots_close_nuc[["POSITION_T","Z", "Y", "X"]].to_numpy()


N_spots_per_single_nuc_T_smooth=np.zeros((N_tp))
for i in range(N_tp):
    lower_bound=i-2
    if lower_bound<0:
        lower_bound=0
    higher_bound=i+3
    if higher_bound>349:
        higher_bound=349
    N_spots_time_range=N_spots_per_single_nuc_T[lower_bound:higher_bound]
    mode_tp=stats.mode(N_spots_time_range).mode
    count_tp=stats.mode(N_spots_time_range).count
    if count_tp>1:
        N_spots_per_single_nuc_T_smooth[i]=stats.mode(N_spots_time_range).mode
    else:
        N_spots_per_single_nuc_T_smooth[i]=np.median(N_spots_time_range)





###To be run even if you skip the rest of part 2
points_layer = viewer.add_points(spots_tzyx_nuc, ndim=4, size=200, scale=scale, blending='additive', opacity=0.3) #ndim=4
#viewer.camera.angles = (-0.26571224801734533, -3.2349084850881065, 146.03256463889608)
viewer.camera.zoom=13.479914708057303
viewer.dims.current_step = (175 , 24, 540, 430)



###PART 3 of the script

Cell_ID=737

cur_spots_to_349=viewer.layers["spots_tzyx_nuc"].data.copy()
cur_spots_349=pd.DataFrame(cur_spots_to_349, columns=["T", "Z", "Y", "X"])
cur_spots_349.to_csv(path_out_im + "cur_spots_t349_id"+str(Cell_ID)+".csv")





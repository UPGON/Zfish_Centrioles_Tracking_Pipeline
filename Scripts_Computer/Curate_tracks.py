#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:12:30 2024

@author: floriancurvaia
"""


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
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy import stats
import pickle 
import time


plt.ioff()
#stack = imread("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/e2-1_cells_of_interest_time_registered_3D.tif")

N_tp=350
path_in_C3=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/nuc_seg_npy")
path_in_C2=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/volumes_V2")
path_in_C1=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/volumes_V2")
with btrack.io.HDF5FileHandler(
  '/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_cells_v2-1.h5', 'r', obj_type='obj_type_1'
) as reader:
  tracks = reader.tracks
  objs = reader.objects
im_prefix=""
Cell_ID=540

path_out_im="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/"
path_config=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/")

scale=(0.75, 0.173, 0.173)

#spots_tzyx_nuc_df=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/cur_spots_t349_id737_v2.csv")
spots_tzyx_nuc_df=pd.read_csv(path_out_im+"cur_spots_t349_id"+str(Cell_ID)+".csv")
spots_tzyx_nuc_df=spots_tzyx_nuc_df.sort_values("T")
spots_tzyx_nuc_df.drop("Unnamed: 0", axis=1, inplace=True)
spots_tzyx_nuc=spots_tzyx_nuc_df[["T", "Z", "Y", "X"]].to_numpy()
spots_nuc_npy_coords=spots_tzyx_nuc_df[["T","Z", "Y", "X"]].astype(int)
spots_tzyx_nuc_df_to_track=spots_tzyx_nuc_df.copy()
spots_tzyx_nuc_df_to_track[["Z", "Y", "X"]]=spots_tzyx_nuc_df_to_track[["Z", "Y", "X"]]*scale

#corner_spots_tzyx_nuc_df=pd.read_csv("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/cur_corner_spots_t126-349_id737.csv")
#Corner_spots=corner_spots_tzyx_nuc_df[["T","Z", "Y", "X"]].to_numpy()


#path_out=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/")


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

scale=(0.75, 0.173, 0.173)

scale_w_T=(1, 0.173, 0.173, 0.75)

xlim=stack_C1.shape[-1]
ylim=stack_C1.shape[-2]
zlim=stack_C1.shape[1]

with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  tracker.configure(path_config /'cell_config.json')

  # append the objects to be tracked
  tracker.append(objs)

  # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
  #tracker.volume = ((0, 832), (0, 911), (0, 67))
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
nuc_coords.to_csv(path_out_im+"nuc_coords_id"+str(Cell_ID)+".csv")

dict_ids_to_track_737 = {
    i: (
        (737, 312, 96, 415, 449, 336, 7) if i < 81 or 87 < i < 92 else
        (737, 415, 449, 336, 312, 96, 7) if i < 113 else
        (737, 449, 415, 336, 312, 96, 7)
    )
    for i in range(N_tp)
}



dict_ids_to_track_540 = {
    i: (
        tuple([540]) if i > 115  else
        tuple([262]) if i > 46 else
        tuple([184]) if i > 31 or i==28 else
        tuple([177]) if i > 29  else
        tuple([189]) if i > 28 else
        tuple([177]) if i > 24 else
        tuple([7]) if i > 16 else
        tuple([136]) if i > 7 else
        tuple([9]) if i > 1 else
        tuple([4]) if i > 0 else
        tuple([6])
        
    )
    for i in range(N_tp)
}




nuc_737_coords=[]
for tp in range(N_tp):
    """
    if Cell_ID==737:
        ids_nuc_to_track=dict_ids_to_track_737[tp]
    elif Cell_ID==135:
        ids_nuc_to_track=(135, 5)
    elif Cell_ID==10:
        ids_nuc_to_track=tuple([10])
    elif Cell_ID==540:
        ids_nuc_to_track=dict_ids_to_track_540[tp]
    """
    ids_nuc_to_track=dict_ids_to_track_737[tp]
    filt_coords=nuc_coords.loc[nuc_coords["T"]==int(tp)]
    for i in ids_nuc_to_track:
        coords=filt_coords.loc[filt_coords.ID == i]
        if not coords.empty:
            break
    
    
    fake_tp=tp
    while coords.empty:
        fake_tp-=1
        ids_nuc_to_track=dict_ids_to_track_737[fake_tp]
        filt_coords=nuc_coords.loc[nuc_coords["T"]==int(fake_tp)]
        for i in ids_nuc_to_track:
            coords=filt_coords.loc[filt_coords.ID == i]
            if not coords.empty:
                break
    coords["T"]=tp
    nuc_737_coords.append(coords)

nuc_737_coords=pd.concat(nuc_737_coords, axis=0, ignore_index=True)
nuc_737_coords.sort_values("T", inplace=True)
#nuc_737_coords.to_csv(path_out_im+"single_nuc_coords_id"+str(Cell_ID)+".csv")
"""
new_nuc_coords=[]
for tp in range(N_tp):
    ids_nuc_to_track=(135, 5)
    
    filt_coords=nuc_coords.loc[nuc_coords["T"]==int(tp)].copy()
    nuc_tp_or=nuc_737_coords.loc[nuc_737_coords["T"]==int(tp)].copy()[["Z", "Y", "X"]].to_numpy()
    filt_coords.loc[filt_coords["Z", "Y", "X"]]=filt_coords.loc[filt_coords["Z", "Y", "X"]]-nuc_tp_or
    
    new_nuc_coords.append(filt_coords)

new_nuc_coords=pd.concat(new_nuc_coords, axis=0, ignore_index=True)
new_nuc_coords.sort_values("T", inplace=True)
"""


new_nuc_coords=nuc_coords.copy()
#new_nuc_coords=all_CMs.copy()
tp_origin={}

for tp in range(N_tp):
    #ids_nuc_to_track=dict_ids_to_track_737[tp]
    #ids_nuc_to_track=(135, 5)
    #ids_nuc_to_track=tuple([10])
    ids_nuc_to_track=dict_ids_to_track_737[tp]
    filt_coords=nuc_coords.loc[nuc_coords["T"]==int(tp)]
    #filt_coords=all_CMs.loc[all_CMs["T"]==int(tp)]
    for i in ids_nuc_to_track:
        coords=filt_coords.loc[filt_coords.ID == i]
        if not coords.empty:
            break
    
    
    fake_tp=tp
    while coords.empty:
        fake_tp-=1
        ids_nuc_to_track=dict_ids_to_track_737[fake_tp]
        filt_coords=nuc_coords.loc[nuc_coords["T"]==int(fake_tp)]
        for i in ids_nuc_to_track:
            coords=filt_coords.loc[filt_coords.ID == i]
            if not coords.empty:
                break
    
    tp_origin[tp]=coords.to_numpy()
    
    new_nuc_coords.loc[new_nuc_coords["T"]==tp, ["Z", "Y", "X"]] =  new_nuc_coords.loc[new_nuc_coords["T"]==tp, ["Z", "Y", "X"]].to_numpy() - coords[["Z", "Y", "X"]].to_numpy() #[["Z", "Y", "X"]] spots_tzyx_nuc_df_to_track.loc[spots_tzyx_nuc_df_to_track["T"]==tp][["Z", "Y", "X"]] -
    
X_max_nuc=np.round(new_nuc_coords.X.max())
Y_max_nuc=np.round(new_nuc_coords.Y.max())
Z_max_nuc=np.round(new_nuc_coords.Z.max())

X_min_nuc=np.round(new_nuc_coords.X.min())
Y_min_nuc=np.round(new_nuc_coords.Y.min())
Z_min_nuc=np.round(new_nuc_coords.Z.min())


"""
nuc_540_new_coords=[]
for i in range(N_tp):
    nuc_540_id=dict_ids_to_track_540[i][0]
    nuc_df=new_nuc_coords.loc[(new_nuc_coords["T"]==i) & (new_nuc_coords["ID"]==nuc_540_id)].copy() #
    nuc_540_new_coords.append(nuc_df)

nuc_540_new_coords=pd.concat(nuc_540_new_coords, axis=0, ignore_index=True)
nuc_540_new_coords.sort_values("T", inplace=True)

nuc_540_dist=[]
for i in range(1, N_tp):
    nuc_540_id=dict_ids_to_track_540[i][0]
    nuc_tp=nuc_540_new_coords.loc[nuc_540_new_coords["T"]==i, ["Z", "Y", "X"]].copy().to_numpy() #
    nuc_m1=nuc_540_new_coords.loc[nuc_540_new_coords["T"]==i-1, ["Z", "Y", "X"]].copy().to_numpy() #
    dist=np.sqrt(np.sum((nuc_tp-nuc_m1)**2))
    
    nuc_540_dist.append(dist)

nuc_540_dist=np.array(nuc_540_dist)
"""

new_nuc_objs=btrack.io.localizations_to_objects(localizations=new_nuc_coords[["T","X", "Y", "Z"]].to_numpy()) #/scale_w_T 

with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  #tracker.configure(path_config /'cell_config.json')
  
  tracker.configure(path_config /'particle_config_2.json')

  # append the objects to be tracked
  tracker.append(new_nuc_objs)

  # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data) in the order X, Y, Z
  #tracker.volume = ( (-40, 9), (-10, 30),(-15, 35))
  tracker.volume = ( (X_min_nuc, X_max_nuc), (Y_min_nuc, Y_max_nuc), (Z_min_nuc, Z_max_nuc))

  # track them (in interactive mode)
  tracker.track_interactive(step_size=10)

  # generate hypotheses and run the global optimizer
  tracker.optimize()

  # store the data in an HDF5 file
  #tracker.export( path_in_spots / 'tracks.h5', obj_type='obj_type_1')

  # get the tracks as a python list
  tracks_spots = tracker.tracks

  # optional: get the data in a format for napari
  data_new, properties_new, graph_new = tracker.to_napari()
  
  tracker.export(path_out_im+'btrack_new_nuc_loc_particle_test.h5', obj_type='obj_type_1')
  

new_nuc_track_coords=pd.DataFrame(data_new, columns=["ID", "T", "Z", "Y", "X"])
for tp in range(N_tp):
    
    origin=tp_origin[tp]
    
    new_nuc_track_coords.loc[new_nuc_track_coords["T"]==tp, ["Z", "Y", "X"]] =  new_nuc_track_coords.loc[new_nuc_track_coords["T"]==tp, ["Z", "Y", "X"]].to_numpy() + origin.flatten()[2:] #origin[["Z", "Y", "X"]].to_numpy()

new_nuc_track_coords.to_csv(path_out_im+"new_all_nuc_coords_tracks_id"+str(Cell_ID)+".csv")
data_new=new_nuc_track_coords.to_numpy()



for tp in range(N_tp):
    
    origin=tp_origin[tp]
    spots_tzyx_nuc_df_to_track.loc[spots_tzyx_nuc_df_to_track["T"]==tp, ["Z", "Y", "X"]] = spots_tzyx_nuc_df_to_track.loc[spots_tzyx_nuc_df_to_track["T"]==tp, ["Z", "Y", "X"]].to_numpy() - origin.flatten()[2:] #[["Z", "Y", "X"]] 


spots_objs=btrack.io.localizations_to_objects(localizations=spots_tzyx_nuc_df_to_track[["T","X", "Y", "Z"]].to_numpy()) #/scale_w_T

#spots_objs=btrack.io.segmentation_to_objects(segmentation=stack_C4, intensity_image=stack_C1, scale=scale)
"""
with btrack.io.HDF5FileHandler(
  '/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_centrioles_id737_loc_seg_test.h5', 'r', obj_type='obj_type_1' #'/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_centrioles_id737.h5'
) as reader:
  spots_objs = reader.objects
"""

X_max_nuc=np.round(spots_tzyx_nuc_df_to_track.X.max())
Y_max_nuc=np.round(spots_tzyx_nuc_df_to_track.Y.max())
Z_max_nuc=np.round(spots_tzyx_nuc_df_to_track.Z.max())

X_min_nuc=np.round(spots_tzyx_nuc_df_to_track.X.min())
Y_min_nuc=np.round(spots_tzyx_nuc_df_to_track.Y.min())
Z_min_nuc=np.round(spots_tzyx_nuc_df_to_track.Z.min())

with btrack.BayesianTracker() as tracker:

  # configure the tracker using a config file
  #tracker.configure(path_config /'cell_config.json')
  
  tracker.configure(path_config /'particle_config_2.json')

  # append the objects to be tracked
  tracker.append(spots_objs)

  # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
  #tracker.volume = ((-15, 35), (-10, 10), (-9, 9))
  tracker.volume = ((-20, 20), (-20, 10), (-13, 9))

  # track them (in interactive mode)
  tracker.track_interactive(step_size=10)

  # generate hypotheses and run the global optimizer
  tracker.optimize()

  # store the data in an HDF5 file
  #tracker.export( path_in_spots / 'tracks.h5', obj_type='obj_type_1')

  # get the tracks as a python list
  tracks_spots = tracker.tracks

  # optional: get the data in a format for napari
  data_spots, properties_spots, graph_spots = tracker.to_napari()
  
  #tracker.export('/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_centrioles_id540_loc_test.h5', obj_type='obj_type_1')
  

spots_track_coords=pd.DataFrame(data_spots, columns=["ID", "T", "Z", "Y", "X"])

for tp in range(N_tp):
    
    origin=tp_origin[tp]
    
    spots_track_coords.loc[spots_track_coords["T"]==tp, ["Z", "Y", "X"]] =  spots_track_coords.loc[spots_track_coords["T"]==tp, ["Z", "Y", "X"]].to_numpy() + origin.flatten()[2:] #.flatten()[2:] #origin[["Z", "Y", "X"]].to_numpy()

new_IDs = np.array([np.where(np.unique(spots_track_coords.ID)==i)[0][0] + 1 for i in spots_track_coords.ID.to_numpy()])
graph_spots_bis={}
unique_spots_IDS=np.unique(spots_track_coords.ID)
for k, v in graph_spots.items():
    k_bis=np.where(unique_spots_IDS==k)[0][0] + 1
    v_bis=np.where(unique_spots_IDS==v)[0][0] + 1
    graph_spots_bis[k_bis]=v_bis
graph_spots = graph_spots_bis
spots_track_coords.ID=new_IDs
spots_track_coords.ID=spots_track_coords.ID.astype(int)
#data_spots[:,0]=spots_track_coords.ID.to_numpy()
data_spots=spots_track_coords.to_numpy()
##Please comment out the line below once the script has been executed once to avoid overwriting the numpy array in the future.
np.save(path_out_im+'spots_coords_particle_track_id'+str(Cell_ID)+'.npy', data_spots)

Recover_data=False

if Recover_data is True:
    data_spots=pd.read_csv(path_out_im + "cur_spots_t349_id"+str(Cell_ID)+".csv")[["ID", "T", "Z", "Y", "X"]].to_numpy()
    new_scale=(1, 0.75, 0.173, 0.173)
    spots_tzyx_nuc=data_spots[:, 1:]/new_scale
    
    data_spots=np.load(path_out_im+'spots_coords_particle_track_id'+str(Cell_ID)+'.npy')
    graph_spots={}
    properties_spots={}

viewer = napari.Viewer(ndisplay=2)


image_layer = viewer.add_image(stack_C1, scale=scale, colormap='green', blending='additive',visible=True, contrast_limits=[90, 400])

#image_layer = viewer.add_image(stack_C1, scale=scale, colormap='green', blending='additive',visible=True)

image_layer = viewer.add_image(stack_C2, scale=scale, colormap='magenta', blending='additive',visible=True, contrast_limits=[90, 500])


viewer.add_tracks(data_new, properties=properties_new, graph=graph_new, visible=True, colormap="turbo", blending="additive") # scale=scale



#image_layer = viewer.add_labels(stack_C4, scale=scale, blending='additive',visible=True)
viewer.add_tracks(data, properties=properties, graph=graph, visible=True, colormap="turbo", blending="additive") # scale=scale


viewer.add_tracks(data_spots, properties=properties_spots, graph=graph_spots, visible=True, colormap="turbo", blending="additive") # scale=scale

points_layer = viewer.add_points(spots_tzyx_nuc, ndim=4, size=200, scale=scale, blending='additive', opacity=0.3) #ndim=4

#points_layer = viewer.add_points(Corner_spots, ndim=4, size=200, scale=scale, blending='additive', opacity=0.3, face_color="yellow") #ndim=4
#mask=points_layer.to_masks(stack_C1.shape)
#viewer.camera.angles = (-0.26571224801734533, -3.2349084850881065, 146.03256463889608)



viewer.camera.zoom=13.479914708057303

viewer.dims.current_step = (0 , 24, 540, 430)

Make_movie=False

if Make_movie is True:
    def center_camera_on_object():
        #ids_nuc_to_track=(135, 5)
        #ids_nuc_to_track=(27, 4) #Non-muscle_cell
        #ids_nuc_to_track=[10]
        #ids_to_track=[]
        tp = viewer.dims.current_step[0]
        ids_nuc_to_track=dict_ids_to_track_737[tp]
        """
        if tp<81 or 87<tp<92: 
            ids_nuc_to_track=(737, 312, 96, 415, 449, 336, 7)
        elif tp<113:
            ids_nuc_to_track=(737, 415, 449, 336, 312, 96, 7)
        else:
            ids_nuc_to_track=(737, 449, 415, 336, 312, 96, 7)
        """
        filt_coords=nuc_coords.loc[nuc_coords["T"]==int(tp)]
        for i in ids_nuc_to_track:
            coords=filt_coords.loc[filt_coords.ID == i]
            if not coords.empty:
                break
        fake_tp=tp
        while coords.empty:
            fake_tp-=1
            ids_nuc_to_track=dict_ids_to_track_737[fake_tp]
            filt_coords=nuc_coords.loc[nuc_coords["T"]==int(fake_tp)]
            for i in ids_nuc_to_track:
                coords=filt_coords.loc[filt_coords.ID == i]
                if not coords.empty:
                    break
            
        viewer.camera.center=(coords.Z.values[0], coords.Y.values[0], coords.X.values[0])
        viewer.dims.current_step = (tp , coords.Z.values[0], coords.Y.values[0], coords.X.values[0])
        
    
    #viewer.dims.events.current_step.connect(center_camera_on_object)
    
    animation = Animation(viewer)
    #viewer.camera.angles=(0.1597470119177224, -3.358462402302114, 140.39807369447243)
    
    viewer.dims.current_step = (0 , 24, 540, 430)
    animation.capture_keyframe(steps=10)
    
    viewer.dims.events.current_step.connect(center_camera_on_object)
    viewer.dims.current_step = (349, 24, 540, 430)
    animation.capture_keyframe(steps=349)
    
    #animation.capture_keyframe(steps=60)
    
    
    #Please write path and filenmae in which to output the animation
    animation.animate(path_out_im+"Centrioles_trackID_And_nuc_id"+str(Cell_ID)+"_topview.mp4", canvas_only=True)



if False is True:
    cur_spots_to_349=viewer.layers["spots_tzyx_nuc"].data.copy()
    cur_spots_349=pd.DataFrame(cur_spots_to_349, columns=["T", "Z", "Y", "X"])
    cur_spots_349.to_csv(path_out_im+"cur_spots_t349_id"+str(Cell_ID)+".csv")
    #cur_spots_349.to_csv(path_out_im + "cur_spots_t155-349_id737.csv")
    Track_ID_of_spot_to_add=12
    spots_track_coords.loc[len(spots_track_coords)] = np.append(Track_ID_of_spot_to_add, cur_spots_to_349[-1]*new_scale)
    spots_track_coords.sort_values(["ID", "T"], inplace=True)
    spots_track_coords.ID=spots_track_coords.ID.astype(int)
    data_spots=spots_track_coords.to_numpy()
    
    np.save(path_out_im+'spots_coords_particle_track_id'+str(Cell_ID)+'.npy', data_spots)







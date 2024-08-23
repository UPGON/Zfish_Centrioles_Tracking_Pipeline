#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:25:25 2024

@author: floriancurvaia
"""


import btrack

from pathlib import Path
from tifffile import imread
import numpy as np
import dask.array as da
from tifffile import imwrite 

#stack = imread("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/e2-1_cells_of_interest_time_registered_3D.tif")


#path_in_C3=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/e2-1_time_registration/3D/Nuc_seg/Tif")
#path_out=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/")
#path_in_C2=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/e2-1_time_registration/3D/volumes")
#path_in_C1=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/e2-1_time_registration/3D/volumes")

path_config=Path("/home/curvaia/btrack_models/")

"""
path_in_C3=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/V2/volumes/npy_seg/")
path_in_C2=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/V2/volumes/")
path_in_C1=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/V2/volumes/")
"""

path_in_C3=Path("/scratch/curvaia/Transplants_e1_2/Muscles_part2/npy_seg/")
path_in_C2=Path("/scratch/curvaia/Transplants_e1_2/Muscles_part2/volumes/")
path_in_C1=Path("/scratch/curvaia/Transplants_e1_2/Muscles_part2/volumes/")

path_out=Path("/scratch/curvaia/Transplants_e1_2/")

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

objs=btrack.io.segmentation_to_objects(segmentation=stack_C3, intensity_image=stack_C2, scale=scale)
"""
with btrack.io.HDF5FileHandler(
  '/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/btrack_cells_v2.h5', 'r', obj_type='obj_type_1'
) as reader:
  #tracks = reader.tracks
  objs = reader.objects
"""
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
  
  tracker.export('/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/btrack_cells_Muscle_v2.h5', obj_type='obj_type_1')

np.save("/scratch/curvaia/Transplants_e1_2/Muscles_part2/nuc_coords_Muscles_V2.npy", data)
  
"""
np.min((data[:,2]/0.75).astype(int))

np.max((data[:,2]/0.75).astype(int))

for nuc_coords in data:
    ID, T, Z, Y, X = nuc_coords
    if int(Z/0.75)>66:
        print(nuc_coords)
        continue
    time_frame_ID= stack_C3[int(T), int(Z/0.75), int(Y/0.173), int(X/0.173)]
    stack_C3[T] [stack_C3[T]==time_frame_ID.compute()] = ID
    
    

imwrite(path_out / ("Nuc_seg_time_track.tif"), stack_C3, imagej=True, metadata={'axes': 'TZYX'})
"""
    
  
  






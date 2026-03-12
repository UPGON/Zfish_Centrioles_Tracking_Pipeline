#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:42:35 2024

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
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from sklearn.linear_model import LinearRegression
import random as rdm

plt.ioff()
#stack = imread("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/e2-1_cells_of_interest_time_registered_3D.tif")
Cell_ID=540
Muscle_cells=True
Open_napari=True
Record_Movie=False
Z_corr=False
log_corr=False


scale=(0.75, 0.173, 0.173)
new_scale=(1, 0.75, 0.173, 0.173)
N_unique_spots=12


Number_of_plots_in_Width=3
Number_of_plots_in_Heigth=4
Size_of_subplots_in_Width=16
Size_of_subplots_in_Heigth=6

Size_of_subplots_in_Width_2=24
Size_of_subplots_in_Heigth_2=48

   
#path_in_C3=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/To_observe_live_transplants/Tif/nuc_seg_npy/")
path_in_C5=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/Spots_seg_trackID_id"+str(Cell_ID)+"_final")
#path_in_C4=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/Spots_seg_id737_V2")
path_in_C2=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/volumes_V2")
path_in_C1=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/Muscle_cells/volumes_V2")
path_out_movies="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Movies/"
with btrack.io.HDF5FileHandler(
  '/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Cluster/btrack_cells_v2-1.h5', 'r', obj_type='obj_type_1'
) as reader:
  tracks = reader.tracks
  objs = reader.objects
im_prefix=""
    


 path_out_im="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Fluo_V2/" #Z_corr/
 if Z_corr:
     path_out_im="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Fluo_V2/Z_corr/" 

    
path_in_files="/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing/"

path_config=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Images/Live_transplants/")


spots_track_coords=pd.read_csv(path_in_files + "all_cur_spots_id"+str(Cell_ID)+"_w_dist.csv") #_w_dist
spots_track_coords.drop("Unnamed: 0", axis=1, inplace=True)

spots_tzyx_nuc=spots_track_coords[["ID", "T", "Z", "Y", "X"]].to_numpy()[:, 1:]/new_scale

spots_nuc_npy_coords=spots_track_coords[["ID","T","Z", "Y", "X"]].astype(int)

if Muscle_cells==True:
    corner_spots_tzyx_nuc_df=pd.read_csv(path_in_files+"all_cur_Corner_spots_id"+str(Cell_ID)+".csv")
    Corner_spots=corner_spots_tzyx_nuc_df[["T","Z", "Y", "X"]].to_numpy()/new_scale


spots_merge=pd.read_csv(path_in_files+"all_cur_spots_id"+str(Cell_ID)+"_w_Merge.csv")
spots_merge.drop("Unnamed: 0", axis=1, inplace=True)


def make_fig_1(N_W_plots=Number_of_plots_in_Width, N_H_plots=Number_of_plots_in_Heigth, 
               scale_width=Size_of_subplots_in_Width, scale_heigth=Size_of_subplots_in_Heigth, 
               N_spots=N_unique_spots, share_x=True, share_y=True):
    
    fig_height=scale_heigth*N_H_plots
    fig_width=scale_width*N_W_plots
    fig, axs=plt.subplots(N_H_plots, N_W_plots, figsize=(fig_width, fig_height), sharex=share_x, sharey=share_y)
    
    if N_W_plots* N_H_plots > N_spots:
        N_diff=N_W_plots* N_H_plots-N_spots
        for i in range(1, N_diff+1):
            fig.delaxes(axs[N_H_plots-1, -i])
    
    elif N_unique_spots==1:
        axs=np.array([axs])

    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    return fig, axs

def make_fig_2(scale_width=Size_of_subplots_in_Width_2, scale_heigth=Size_of_subplots_in_Heigth_2, 
               N_spots=N_unique_spots, share_x=True, share_y=True):
    N_W_plots=2
    if not N_spots % 2 == 0 and N_spots!=1:
       N_H_plots=N_spots//2 +1
       
    elif N_spots==1:
       N_H_plots=1
       N_W_plots=1

    else:
       N_H_plots=N_spots//2
    fig_height=scale_heigth*N_H_plots
    fig_width=scale_width*N_W_plots
    fig, axs=plt.subplots(N_H_plots, N_W_plots, figsize=(fig_width, fig_height), sharex=share_x, sharey=share_y)
    
    if not N_spots % 2 == 0 and N_spots!=1:
        fig.delaxes(axs[N_H_plots-1, -1])
    
    elif N_unique_spots==1:
        axs=np.array([axs])

    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    return fig, axs

#path_out=Path("/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/")


#nuc_seg_file="/Volumes/users/curvaia/Images/Live/20240321_Transplants/20240321_172724_Transplants_TgCentrinEos_H2BmCherry/e2-1_FLUO/Nuc_seg_time_track.tif"
#path_in_C3=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/tif_seg/")
#path_in_C2=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/")
#path_in_C1=Path("/scratch/curvaia/Transplants_e1_2/Cells_of_interest_3D/")

#path_out=Path("/scratch/curvaia/Transplants_e1_2/")

#filenames_C4 = sorted(path_in_C4.glob('*.npy'))

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
"""
def read_one_image_C4(block_id, filenames=filenames_C4, axis=0):
    # a function that reads in one chunk of data
    path = filenames[block_id[axis]]
    image = np.load(path)
    return np.expand_dims(image, axis=axis)

"""



sample_C2 = imread(filenames_C2[0]) #np.transpose(imread(filenames_C2[0]), (2,1,0))

sample_C1 = imread(filenames_C1[0])

#sample_C4 = np.load(filenames_C4[0])

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
"""
stack_C4 = da.map_blocks(
    read_one_image_C4,
    dtype=sample_C4.dtype,
    chunks=( (1,) * len(filenames_C4), *sample_C4.shape )
)
"""
stack_C1=stack_C1.astype("int16")
N_tp=stack_C1.shape[0]

path_out_im= path_out_im +"mean_fluo_vs_time_id"+str(Cell_ID)+"/"
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

dict_ids_to_track_737 = {
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
#Cell_ID=540
for tp in range(N_tp):
    
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

    
all_CMs=pd.read_csv(path_in_files+"All_CMs_id"+str(Cell_ID)+".csv")
all_CMs.drop("Unnamed: 0", axis=1, inplace=True)

graph_spots = {}
properties_spots={}
data_spots=spots_track_coords[["ID","T","Z", "Y", "X"]].to_numpy()


#np.save(path_out_im+'spots_coords_track_id737.npy', data_spots)

Unique_spots=list(range(1, N_unique_spots+1)) #Change between 6 (737) and 2 (135)
all_spots_w_fluo=[]
single_spots_df_all_tp={}
for i in Unique_spots:
    spot_df=spots_track_coords.loc[spots_track_coords["ID"]==i].copy()
    spot_df.sort_values("T", inplace=True)
    spot_df.reset_index(inplace=True, drop=True)
    all_tp=set(range(N_tp))
    present_tp=set(spot_df["T"])
    origin_737=nuc_737_coords.loc[nuc_737_coords["T"].isin(present_tp), ["Z", "Y", "X"]]
    spot_df[["Z_reg", "Y_reg", "X_reg"]]=spot_df[["Z", "Y", "X"]]-origin_737.to_numpy()
    all_spots_w_fluo.append(spot_df)
    missing_tp=all_tp-present_tp
    all_tp_spot_df=spot_df.copy()
    for t in missing_tp:
        #if Fluo_V2:
        #    new_row=np.empty(12)
        #else:
        new_row=np.empty(10)
        new_row[:]=np.nan
        new_row[0]=i
        new_row[1]=t
        all_tp_spot_df.loc[len(all_tp_spot_df)] = new_row
        #spot_df
    all_tp_spot_df.sort_values("T", inplace=True)
    all_tp_spot_df.reset_index(inplace=True, drop=True)
    single_spots_df_all_tp[i]=all_tp_spot_df

spots_track_coords_w_fluo=pd.concat(all_spots_w_fluo, axis=0, ignore_index=True)

spots_track_coords_w_fluo.sort_values(["T", "ID"], inplace=True)



with open(path_in_files+'dict_spots_fluo_values_r2_5_xy0_75_z1_66_per_tp_id'+str(Cell_ID)+'.pkl', 'rb') as f:
    spots_fluo_val_tp = pickle.load(f)


all_spots_w_fluo=[]
spots_track_coords_w_fluo["Fluo"]=np.nan

spots_mean_fluo_tp={}
for s in Unique_spots:
   #spots_mean_fluo_tp[s]=np.empty((N_tp))
   spots_mean_fluo_tp[s]=[]
dict_spot_size_tp={}
for s in Unique_spots:
    dict_spot_size_tp[s]=np.empty((N_tp))
    dict_spot_size_tp[s][:]=np.nan



for i in range(N_tp):
    spots_df_tp=spots_track_coords_w_fluo.loc[spots_track_coords_w_fluo["T"]==i].copy()
    unique_spots_tp=np.unique(spots_df_tp.ID.to_numpy())
    for s in unique_spots_tp:
        values=spots_fluo_val_tp[i][s]
        q=np.max(values)-np.std(values)*1.15
        q=np.mean(values)+np.std(values)*1.15
        #q=np.mean(values)+np.std(values)*1.75
        vals_to_keep=values[values>q]
        dict_spot_size_tp[s][i]=len(vals_to_keep)
        #fluo_v2=np.sum(vals_to_keep)
        spots_mean_fluo_tp[s].append(np.mean(vals_to_keep))
        fluo_v2=np.mean(vals_to_keep)
        if Z_corr:
            if log_corr:
                corr=np.log(spots_df_tp.loc[spots_df_tp.ID==s, "Z"].values[0]+1)
            else:
                corr=spots_df_tp.loc[spots_df_tp.ID==s, "Z"].values[0]+1
        else:
            corr=1
        spots_df_tp.loc[spots_df_tp.ID==s, "Fluo"]=fluo_v2#/corr
    all_spots_w_fluo.append(spots_df_tp)


spots_track_coords_w_fluo=pd.concat(all_spots_w_fluo, axis=0, ignore_index=True)

spots_track_coords_w_fluo.sort_values(["T", "ID"], inplace=True)


if Z_corr:
    min_max_spot_dict={}
    for s in Unique_spots:
        spot_df=spots_track_coords_w_fluo.loc[spots_track_coords_w_fluo["ID"]==s].copy()
        fluo_spot=spot_df["Fluo"]
        min_max_spot_dict[s]=[fluo_spot.min(), fluo_spot.max()]
        #fluo_spot=(fluo_spot-fluo_spot.mean())/fluo_spot.std()
        fluo_spot=(fluo_spot-fluo_spot.min())/(fluo_spot.max()-fluo_spot.min())
        
        spots_track_coords_w_fluo.loc[spots_track_coords_w_fluo["ID"]==s, "Fluo"]=fluo_spot
    
    path_corr=spots_track_coords_w_fluo["Z"].to_numpy().reshape(-1, 1) *-1
    path_corr-=path_corr.min()
    intensity_plot=spots_track_coords_w_fluo["Fluo"].to_numpy().reshape(-1, 1)
    intensity_reg=spots_track_coords_w_fluo["Fluo"].to_numpy().reshape(-1, 1)
    fig, ax=plt.subplots()
    ax.scatter(path_corr,intensity_plot, c=spots_track_coords_w_fluo["T"], s=1)
    ax.set_xlabel("Z")
    ax.set_ylabel("Fluo")
    fig.savefig(path_out_im+im_prefix+"All_Z_vs_Fluo_ColTime.png", dpi=300, bbox_inches='tight')
    plt.close()


    #path_corr=spots_track_coords_w_fluo["Z"].to_numpy().reshape(-1, 1)
    #intensity=spots_track_coords_w_fluo["Fluo_V2"].to_numpy().reshape(-1, 1)
    time=spots_track_coords_w_fluo["T"].to_numpy().reshape(-1, 1)
    fig, ax=plt.subplots()
    ax.scatter(path_corr,intensity_plot , s=1)
    #ax.scatter(medium_path, np.log(feat_filt_all[stain]), s=1, alpha=0.1)
    """
    regressor = LinearRegression()
    regressor.fit(path_corr, np.log(intensity_reg+1)) #
    normalizer_stain = np.exp(regressor.predict([[0]])) / np.exp( #
        regressor.predict(path_corr)) #
    #joblib.dump(regressor, path_in_df+stain+"_exp_regressor.sav")
    """
    """
    regressor = LinearRegression()
    regressor.fit(path_corr, intensity_reg) #
    normalizer_stain = regressor.predict([[0]]) /( #
        regressor.predict(path_corr)) #
    """
    
    if log_corr:
        regressor = LinearRegression()
        regressor.fit(np.stack([path_corr.flatten(), time.flatten()]).T, np.log(intensity_reg+1)) #, 'embryo_path'
        normalizer_stain = np.exp(regressor.predict([[0, 0]])) /np.exp( #, 0
           regressor.predict(np.stack([path_corr.flatten(), time.flatten()]).T))

    else:
        regressor = LinearRegression()
        regressor.fit(np.stack([path_corr.flatten(), time.flatten()]).T, intensity_reg) #, 'embryo_path'
        normalizer_stain = regressor.predict([[0, 0]]) /( #, 0
           regressor.predict(np.stack([path_corr.flatten(), time.flatten()]).T))
    
    ax.scatter(path_corr, intensity_plot*normalizer_stain, s=1, alpha=0.1)
    ax.set_xlabel("Z")
    ax.set_ylabel("Fluo")
    fig.savefig(path_out_im+im_prefix+"Z_corr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    spots_track_coords_w_fluo["Fluo"]=spots_track_coords_w_fluo["Fluo"]*normalizer_stain.flatten()
    for s in Unique_spots:
        spot_df=spots_track_coords_w_fluo.loc[spots_track_coords_w_fluo["ID"]==s].copy()
        fluo_spot=spot_df["Fluo"]
        #fluo_spot=(fluo_spot-fluo_spot.mean())/fluo_spot.std()
        fluo_spot_min, fluo_spot_max=min_max_spot_dict[s]
        fluo_spot=(fluo_spot*(fluo_spot_max-fluo_spot_min))+fluo_spot_min
        spots_track_coords_w_fluo.loc[spots_track_coords_w_fluo["ID"]==s, "Fluo"]=fluo_spot

    intensity=spots_track_coords_w_fluo["Fluo"].to_numpy().reshape(-1, 1)
    fig, ax=plt.subplots()
    ax.scatter(path_corr,intensity, c=spots_track_coords_w_fluo["T"], s=1)
    ax.set_xlabel("Z")
    ax.set_ylabel("Fluo")
    fig.savefig(path_out_im+im_prefix+"All_Z_vs_Fluo_ColTime_Corr.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax=plt.subplots()
    ax.scatter(path_corr,intensity, c=spots_track_coords_w_fluo["ID"], s=1)
    ax.set_xlabel("Z")
    ax.set_ylabel("Fluo")
    fig.savefig(path_out_im+im_prefix+"All_Z_vs_Fluo_ColSpotID_Corr.png", dpi=300, bbox_inches='tight')
    plt.close()

else:
    path_corr=spots_track_coords_w_fluo["Z"].to_numpy().reshape(-1, 1) *-1
    path_corr-=path_corr.min()
    intensity_plot=spots_track_coords_w_fluo["Fluo"].to_numpy().reshape(-1, 1)
    intensity_reg=spots_track_coords_w_fluo["Fluo"].to_numpy().reshape(-1, 1)
    fig, ax=plt.subplots()
    ax.scatter(path_corr,intensity_plot, c=spots_track_coords_w_fluo["T"], s=1)
    ax.set_xlabel("Z")
    ax.set_ylabel("Fluo")
    fig.savefig(path_out_im+im_prefix+"All_Z_vs_Fluo_ColTime.png", dpi=300, bbox_inches='tight')
    plt.close()


single_spots_df_all_tp={} 

for i in Unique_spots:
    spot_df=spots_track_coords_w_fluo.loc[spots_track_coords_w_fluo["ID"]==i].copy()
    spot_df.sort_values("T", inplace=True)
    spot_df.reset_index(inplace=True, drop=True)
    all_tp=set(range(N_tp))
    present_tp=set(spot_df["T"])
    missing_tp=all_tp-present_tp
    all_tp_spot_df=spot_df.copy()
    for t in sorted(missing_tp):
        #if Fluo_V2:
        #    new_row=np.empty(12)
        #else:
        new_row=np.empty(11)
        new_row[:]=np.nan
        new_row[0]=i
        new_row[1]=t
        all_tp_spot_df.loc[len(all_tp_spot_df)] = new_row
        #print(len(all_tp_spot_df))
        #spot_df
    all_tp_spot_df.sort_values("T", inplace=True)
    all_tp_spot_df.reset_index(inplace=True, drop=True)
    single_spots_df_all_tp[i]=all_tp_spot_df


def plot_colourline(x,y,c, ax):
    col = cm.turbo((c-np.min(c))/(np.max(c)-np.min(c)))
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=col[i]) #,  vmin=0, vmax=N_tp-1
    im = ax.scatter(x, y, c=c, s=0, cmap=cm.turbo,  vmin=0, vmax=N_tp-1)
    return im

fig, axs=make_fig_1()

for i, ax in zip(Unique_spots, axs.flatten()):
    spot_df=single_spots_df_all_tp[i]
    ax.plot(spot_df["T"], spot_df["Fluo"]) #, c=spot_df["T"]
    ax.set_ylabel("Mean fluo intensity for spot "+str(i))
    ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_Mean_fluo_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, axs=make_fig_1()

for i, ax in zip(Unique_spots, axs.flatten()):
    spot_df=single_spots_df_all_tp[i]
    #ax.plot(spot_df["T"], spot_df[Fluo]) #, c=spot_df["T"]
    plot_colourline(spot_df["T"], spot_df["Fluo"], spot_df["Z"], ax)
    ax.set_ylabel("Mean fluo intensity for spot "+str(i))
    ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_Mean_fluo_per_spots_vs_time_ColorZ.png", bbox_inches='tight', dpi=300)
plt.close()


fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.075)
fig.subplots_adjust(wspace=0.075)
for i in Unique_spots:
    spot_df=single_spots_df_all_tp[i]
    ax.plot(spot_df["T"], spot_df["Fluo"])
ax.set_ylabel("Mean fluo intensity per spot")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"all_Mean_fluo_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.075)
fig.subplots_adjust(wspace=0.075)
for i in Unique_spots:
    spot_df=single_spots_df_all_tp[i]
    ax.plot(spot_df["T"], spot_df["Dist_nuc"])
ax.set_ylabel("Distance to closest Nucleus")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"all_dist_nuc_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, axs=make_fig_1()

for i, ax in zip(Unique_spots, axs.flatten()):
    spot_df=single_spots_df_all_tp[i]
    ax.plot(spot_df["T"], spot_df["Dist_nuc"])
    ax.set_ylabel("Distance to closest Nucleus (spot "+str(i)+")")
    ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_dist_nuc_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, axs=make_fig_1(share_y=False)
    
fig.subplots_adjust(hspace=0.075)
fig.subplots_adjust(wspace=0.125)
for i, ax in zip(Unique_spots, axs.flatten()):
    spot_df=single_spots_df_all_tp[i]
    ax.scatter(spot_df["Z"], spot_df["Fluo"], c=spot_df["T"], cmap="viridis", s=5)
    ax.set_ylabel("Mean Intensity (spot "+str(i)+")")
    ax.set_xlabel("Z")
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.125, hspace=0.075)
fig.savefig(path_out_im+im_prefix+"single_fluo_spots_vs_Z.png", bbox_inches='tight', dpi=300)
plt.close()

fig, axs=make_fig_1()
for i, ax in zip(Unique_spots, axs.flatten()):
    spot_df=single_spots_df_all_tp[i]
    ax.scatter(spot_df["Z"], spot_df["T"], c=spot_df["Fluo"], cmap="viridis", s=5)
    ax.set_ylabel("T (spot "+str(i)+")")
    ax.set_xlabel("Z")
fig.savefig(path_out_im+im_prefix+"single_T_spots_vs_Z.png", bbox_inches='tight', dpi=300)
plt.close()


fig, axs=make_fig_1()
for i, ax in zip(Unique_spots, axs.flatten()):
    spot_sizes=dict_spot_size_tp[i]
    ax.plot(range(N_tp), spot_sizes)
    ax.set_ylabel("N° of pixels in spot "+str(i))
    ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_spots_size_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()



for i in Unique_spots:
    fig, axs=plt.subplots(2, 2, figsize=(15, 15) ) #sharex=True, sharey=True
    fig.delaxes(axs[1, 1])
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)
    spot_df=single_spots_df_all_tp[i]
    #axs[0,0].plot(spot_df["X_reg"], spot_df["Y_reg"])
    im_xy=plot_colourline(spot_df["X_reg"], spot_df["Y_reg"], spot_df["T"], axs[0,0])
    axs[0,0].set_xlabel("X")
    axs[0,0].set_ylabel("Y")
    #axs[0,1].plot(spot_df["Z_reg"], spot_df["Y_reg"])
    im_zy=plot_colourline(spot_df["Z_reg"], spot_df["Y_reg"], spot_df["T"], axs[0,1])
    axs[0,1].set_xlabel("Z")
    axs[0,1].set_ylabel("Y")
    #axs[1,0].plot(spot_df["X_reg"], spot_df["Z_reg"])
    im_xz=plot_colourline(spot_df["X_reg"], spot_df["Z_reg"], spot_df["T"], axs[1,0])
    axs[1,0].set_xlabel("X")
    axs[1,0].set_ylabel("Z")
    
    fig.savefig(path_out_im+im_prefix+"spot_"+str(i)+"_position_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_colourline_3D(x,y,z, c, ax):
    col = cm.turbo((c-np.min(c))/(np.max(c)-np.min(c)))
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], [z[i],z[i+1]], c=col[i])
    im = ax.scatter(x, y, z, c=c, s=0, cmap=cm.turbo, vmin=0, vmax=N_tp-1)
    return im

for i in Unique_spots:
    fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True, subplot_kw={"projection":"3d"})
    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    spot_df=single_spots_df_all_tp[i]
    #ax.plot(spot_df["X_reg"], spot_df["Y_reg"], spot_df["Z_reg"])
    im_xyz=plot_colourline_3D(spot_df["X_reg"], spot_df["Y_reg"], spot_df["Z_reg"], spot_df["T"], ax)
    ax.set_zlabel("Y")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    fig.savefig(path_out_im+im_prefix+"spot_"+str(i)+"_position_vs_time_3D.png", bbox_inches='tight', dpi=300)
    plt.close()


for i in Unique_spots:
    spot_df=single_spots_df_all_tp[i]
    #ax.plot(spot_df["X_reg"], spot_df["Y_reg"], spot_df["Z_reg"])
    #fig = px.line_3d(spot_df, x='X_reg', y='Y_reg', z='Z_reg', color="T", color_discrete_sequence=px.colors.sequential.Turbo)
    fig = go.Figure(data=go.Scatter3d(
    x=spot_df["X_reg"], y=spot_df["Y_reg"], z=spot_df["Z_reg"],
    marker=dict(
        size=4,
        color=spot_df["T"],
        colorscale='Turbo',
    ),
    line=dict(
        color='darkgrey',
        width=2
    )
    ))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), coloraxis_colorbar_title_text = 'Time')
    fig.write_html(path_out_im+im_prefix+"spot_"+str(i)+"_position_vs_time_3D.html")





for i in np.unique(spots_track_coords.ID):
    fig, ax=plt.subplots(figsize=(10, 5))
    ax.plot(dict_spots_tp[i], spots_mean_fluo_tp[i])
    ax.set_ylabel("Mean fluo intensity per spot")
    ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+im_prefix+"single_Mean_fluo_per_spots_vs_time_ID_"+str(i)+".png", bbox_inches='tight', dpi=300)
    plt.close()

for i in np.unique(spots_track_coords.ID):
    fig, ax=plt.subplots(figsize=(10, 5))
    spot_df=single_spots_df_all_tp[i]
    ax.plot(spot_df["T"], spot_df["Dist_nuc"])
    ax.set_ylabel("Distance to closest Nucleus")
    ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+im_prefix+"single_dist_nuc_per_spots_vs_time_ID_"+str(i)+".png", bbox_inches='tight', dpi=300)
    plt.close()
        

N_spots_per_tp=[]
for i in range(N_tp):
    N_spots_tp=len(spots_track_coords.loc[spots_track_coords["T"]==i])
    N_spots_per_tp.append(N_spots_tp)

N_spots_per_tp=np.array(N_spots_per_tp)


N_spots_per_tp_smooth=np.zeros((N_tp))
for i in range(N_tp):
    lower_bound=i-2
    if lower_bound<0:
        lower_bound=0
    higher_bound=i+3
    if higher_bound>N_tp-1:
        higher_bound=N_tp-1
    N_spots_time_range=N_spots_per_tp[lower_bound:higher_bound]
    mode_tp=stats.mode(N_spots_time_range).mode
    count_tp=stats.mode(N_spots_time_range).count
    if count_tp>1:
        N_spots_per_tp_smooth[i]=stats.mode(N_spots_time_range).mode
    else:
        N_spots_per_tp_smooth[i]=np.median(N_spots_time_range)

N_spots_per_nuc_per_tp=[]

N_spots_per_nuc_per_tp_T=[]

Nuc_IDs_per_tp=[]

mean_spots_dist_nuc_per_tp=[]

N_nucs_per_tp_list=[]

for i in range(N_tp):
    spots_tp=spots_track_coords.loc[spots_track_coords["T"]==i]
    mean_spots_dist_nuc_tp=np.mean(spots_tp["Dist_nuc"])
    mean_spots_dist_nuc_per_tp.append(mean_spots_dist_nuc_tp)
    all_nucs_tp=np.unique(spots_tp["Nuc_ID"])
    N_nucs_per_tp_list.append(len(all_nucs_tp))
    for n in all_nucs_tp:
        N_spots_per_nuc_tp = len(spots_tp.loc[spots_tp["Nuc_ID"]==n])
        N_spots_per_nuc_per_tp.append(N_spots_per_nuc_tp)
        N_spots_per_nuc_per_tp_T.append(i)
        Nuc_IDs_per_tp.append(n)
   
N_nucs_per_tp=np.array(N_nucs_per_tp_list)
if Cell_ID==737:
    N_nucs_per_tp[np.where(N_nucs_per_tp>3)]=3
Mean_N_spots_per_nuc_per_tp=N_spots_per_tp/N_nucs_per_tp
if Cell_ID==737:
    with open(path_in_files+'dict_spots_per_nuc_per_tp_id'+str(Cell_ID)+'.pkl', 'rb') as f:
        dict_spot_to_nuc_per_tp = pickle.load(f)
    
    dict_sing_spots_per_sing_nuc_per_tp={}    
    dict_N_spots_per_sing_nuc_per_tp={}
    dict_sing_nuc_labs_per_tp={}
    
    for tp, spots_and_nuc_tp in dict_spot_to_nuc_per_tp.items():
        count_per_nuc={}
        spots_per_nuc={}
        dict_N_spots_per_sing_nuc_per_tp[tp]={}
        dict_sing_nuc_labs_per_tp[tp]={}
        dict_sing_spots_per_sing_nuc_per_tp[tp]={}
        for spot_ID, nuc_ID in spots_and_nuc_tp.items():
            if nuc_ID in count_per_nuc:
                count_per_nuc[nuc_ID]+=1
                spots_per_nuc[nuc_ID].append(spot_ID)
            else:
                count_per_nuc[nuc_ID]=1
                spots_per_nuc[nuc_ID]=[spot_ID]
                
        nucs_IDs_tp=list(count_per_nuc.keys())
        X_coords_nucs_tp={}
        nucs_tp_coords=nuc_coords.loc[nuc_coords["T"]==tp].copy()
        for nuc_label in nucs_IDs_tp:
            X_coords_nucs_tp[nuc_label]=nucs_tp_coords.loc[nucs_tp_coords["ID"]==nuc_label, "X"].values[0]
        nucs_processed=[]
        nuc_left=min(X_coords_nucs_tp, key=X_coords_nucs_tp.get)
        dict_N_spots_per_sing_nuc_per_tp[tp]["nuc_left"]=count_per_nuc[nuc_left]
        dict_sing_spots_per_sing_nuc_per_tp[tp]["nuc_left"]=spots_per_nuc[nuc_left]
        dict_sing_nuc_labs_per_tp[tp]["nuc_left"]=nuc_left
        nucs_processed.append(nuc_left)
        if len(nucs_IDs_tp)>1:
            nuc_right=max(X_coords_nucs_tp, key=X_coords_nucs_tp.get)
            dict_N_spots_per_sing_nuc_per_tp[tp]["nuc_right"]=count_per_nuc[nuc_right]
            dict_sing_spots_per_sing_nuc_per_tp[tp]["nuc_right"]=spots_per_nuc[nuc_right]
            dict_sing_nuc_labs_per_tp[tp]["nuc_right"]=nuc_right
            nucs_processed.append(nuc_right)
            if len(nucs_IDs_tp)>2:
                nucs_left=set(nucs_IDs_tp)-set(nucs_processed)
                if len(nucs_left)==1:
                    nuc_middle=list(nucs_left)[0]
                    dict_N_spots_per_sing_nuc_per_tp[tp]["nuc_middle"]=count_per_nuc[nuc_middle]
                    dict_sing_spots_per_sing_nuc_per_tp[tp]["nuc_middle"]=spots_per_nuc[nuc_middle]
                    dict_sing_nuc_labs_per_tp[tp]["nuc_middle"]=nuc_middle
                else:
                    nuc_middle=min(nucs_left)
                    count_sp=0
                    sing_spots=[]
                    for nu in nucs_left:
                        count_sp+=count_per_nuc[nu]
                        sing_spots.extend(spots_per_nuc[nu])
                    dict_N_spots_per_sing_nuc_per_tp[tp]["nuc_middle"]=count_sp
                    dict_sing_spots_per_sing_nuc_per_tp[tp]["nuc_middle"]=sing_spots
                    dict_sing_nuc_labs_per_tp[tp]["nuc_middle"]=nuc_middle
         
        
    dict_N_spots_per_sing_nuc={}
    dict_N_spots_per_sing_nuc["nuc_left"]=[]
    dict_N_spots_per_sing_nuc["nuc_middle"]=[]
    dict_N_spots_per_sing_nuc["nuc_right"]=[]
    all_3_nucs=["nuc_left", "nuc_middle", "nuc_right"]
    
    for i in range(N_tp):
        for n in all_3_nucs:
            if n in dict_N_spots_per_sing_nuc_per_tp[i].keys():
                dict_N_spots_per_sing_nuc[n].append(dict_N_spots_per_sing_nuc_per_tp[i][n])
            else:
                dict_N_spots_per_sing_nuc[n].append(0)
                
    fig, ax=plt.subplots()
    for n in all_3_nucs:
        ax.plot(range(N_tp), dict_N_spots_per_sing_nuc[n],  label=n)
        #ax.scatter(range(N_tp), dict_N_spots_per_sing_nuc[n],  label=n, s=5)
    ax.set_ylabel("Mean N° of spots per Nucleus")
    ax.set_xlabel("Time frame (a.u)")
    ax.legend()
    fig.savefig(path_out_im+im_prefix+"N_spots_per_single_nuc_vs_time_nuc.png", bbox_inches='tight', dpi=300)
    plt.close()

"""
with open(path_out_im + "Closest_nuc_IDs_per_spot_per_tp_id"+str(Cell_ID)+".txt", "w") as f:
    for i in range(N_tp):
        f.write("T: "+str(i)+"\n")
        spots_tp=spots_track_coords.loc[spots_track_coords["T"]==i]
        for spot in spots_tp.iterrows():
            f.write(str(int(spot[1]["ID"]))+": "+str(int(spot[1]["Nuc_ID"]))+"\n")
"""

fig, ax=plt.subplots()
#ax.plot(range(N_tp), mean_spots_per_nuc_id135)
ax.stem(range(N_tp), N_spots_per_tp, bottom=2, markerfmt="none")
ax.set_ylabel("N° of spots in cell")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"N_spots_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, ax=plt.subplots()
#ax.plot(range(N_tp), mean_spots_per_nuc_id135)
ax.stem(range(N_tp), N_spots_per_tp_smooth, bottom=2, markerfmt="none")
ax.set_ylabel("N° of spots in cell")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"N_spots_smooth_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, ax=plt.subplots()
ax.scatter(N_spots_per_nuc_per_tp_T, N_spots_per_nuc_per_tp, s=5, marker=".") #, c=Nuc_IDs_per_tp, cmap="tab20"
ax.set_ylabel("N° of spots for each nucleus")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"N_spots_per_nuc_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()


fig, ax=plt.subplots()
ax.scatter(range(N_tp), mean_spots_dist_nuc_per_tp)
ax.set_ylabel("Mean distance between spots and nucleus")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"Mean_dist_spots_nuc_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, ax=plt.subplots()
ax.plot(range(N_tp), N_nucs_per_tp)
ax.set_ylabel("N° of Nuclei")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"N_nuclei_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()


fig, ax=plt.subplots()
ax.plot(range(N_tp), Mean_N_spots_per_nuc_per_tp)
ax.set_ylabel("Mean N° of spots per Nucleus")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"Mean_N_spots_per_nuc_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, ax=plt.subplots()
ax.scatter(mean_spots_dist_nuc_per_tp,Mean_N_spots_per_nuc_per_tp, c=range(N_tp), s=5)
ax.set_ylabel("N° of spots for nucleus")
ax.set_xlabel("Mean distance between spots and nucleus")
fig.savefig(path_out_im+im_prefix+"Mean_spots_per_nuc_vs_Mean_dist_spots_nuc.png", bbox_inches='tight', dpi=300)
plt.close()


spots_fluo_to_df=np.zeros((len(Unique_spots), N_tp))
for s in Unique_spots:
    spot_df=single_spots_df_all_tp[s]
    spots_fluo_to_df[s-1]=spot_df[Fluo].to_numpy()

dict_spots_speed={}
for s in Unique_spots:
    spot_df=single_spots_df_all_tp[s].copy()
    #spot_df["Spot_ID"]=s
    #spots_position_to_df.append(spot_df[["Spot_ID","X_reg", "Y_reg", "Z_reg"]])
    X_pos=spot_df["X_reg"].to_numpy()
    Y_pos=spot_df["Y_reg"].to_numpy()
    Z_pos=spot_df["Z_reg"].to_numpy()
    X_diff=X_pos[1:N_tp]-X_pos[:N_tp-1]
    Y_diff=Y_pos[1:N_tp]-Y_pos[:N_tp-1]
    Z_diff=Z_pos[1:N_tp]-Z_pos[:N_tp-1]
    dist=np.sqrt(X_diff**2+Y_diff**2+Z_diff**2)
    dict_spots_speed[s]=dist

fig, axs=make_fig_1()
for i, ax in zip(Unique_spots, axs.flatten()):
    spot_speed=dict_spots_speed[i]
    ax.plot(range(1, N_tp), spot_speed)
    ax.set_ylabel("spot "+str(i)+" speed (a.u.)")
    ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_spots_speed_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.075)
fig.subplots_adjust(wspace=0.075)
for i in Unique_spots:
    spot_speed=dict_spots_speed[i]
    ax.plot(range(1, N_tp), spot_speed)
ax.set_ylabel("spot "+str(i)+" speed (a.u.)")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"all_spots_speed_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

#spots_position_df=pd.concat(spots_position_to_df, axis=1, ignore_index=True)

corr_method="pearson"

spots_fluo_df=pd.DataFrame(spots_fluo_to_df)
corr_mat_fluo=spots_fluo_df.T.corr(method=corr_method).to_numpy()

spots_diff_fluo_to_df=spots_fluo_to_df[:, :-1]-spots_fluo_to_df[:, 1:]
spots_diff_fluo_df=pd.DataFrame(spots_diff_fluo_to_df)
corr_mat_diff_fluo=spots_diff_fluo_df.T.corr(method=corr_method).to_numpy()

fig, ax=plt.subplots() #figsize=(10,10)
coord_plot=ax.imshow(corr_mat_fluo, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
#ax.set_xlabel("Theta bin")
#ax.set_ylabel("Phi bin")
for i in range(len(Unique_spots)):
    for j in range(len(Unique_spots)):
        cell_value = corr_mat_fluo[i, j]
        cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
        
        # Determine whether the cell color is dark or light
        #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
        luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
        if luminance > 0.5:
            text_color = 'black'  # Use black for light backgrounds
        else:
            text_color = 'white'  # Use white for dark backgrounds
        
        ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                     xy=(j, i),
                     ha='center', va='center',
                     fontsize=7.5,
                     color=text_color)
fig.colorbar(coord_plot, ax=ax, label=corr_method+" correlation coefficient")
ax.set_box_aspect(1)
fig.savefig(path_out_im+im_prefix+"Time_"+corr_method+"_correlation_fluo_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
plt.close()

fig, ax=plt.subplots() #figsize=(10,10)
coord_plot=ax.imshow(corr_mat_diff_fluo, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
#ax.set_xlabel("Theta bin")
#ax.set_ylabel("Phi bin")
for i in range(len(Unique_spots)):
    for j in range(len(Unique_spots)):
        cell_value = corr_mat_diff_fluo[i, j]
        cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
        
        # Determine whether the cell color is dark or light
        #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
        luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
        if luminance > 0.5:
            text_color = 'black'  # Use black for light backgrounds
        else:
            text_color = 'white'  # Use white for dark backgrounds
        
        ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                     xy=(j, i),
                     ha='center', va='center',
                     fontsize=7.5,
                     color=text_color)
fig.colorbar(coord_plot, ax=ax, label=corr_method+" correlation coefficient")
ax.set_box_aspect(1)
fig.savefig(path_out_im+im_prefix+"Time_"+corr_method+"_correlation_diff_fluo_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
plt.close()

cosine_pos_dict={}
for s in Unique_spots:
   cosine_pos_dict[s]=[] 

for s in Unique_spots:
    spot_df=single_spots_df_all_tp[s]
    for t in range(1, N_tp):
        pos_t=spot_df.loc[spot_df["T"]==t, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
        pos_tm1=spot_df.loc[spot_df["T"]==t-1, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
        cosine = np.dot(pos_tm1,pos_t)/(np.linalg.norm(pos_tm1)*np.linalg.norm(pos_t))
        cosine_pos_dict[s].append(cosine)
        
fig, axs=make_fig_1()
for i, ax in zip(Unique_spots, axs.flatten()):
    cosine_spot=cosine_pos_dict[i]
    ax.plot(range(1,N_tp), cosine_spot)
    ax.set_ylabel("Cosine similarity between position vectors (spot "+str(i)+")")
    ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_cosine_similarity_position_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()

Cosine_Nuc_dirs=[]

Vec_Nuc_dirs=[]


for t in range(1, N_tp-1):
    pos_t=nuc_737_coords.loc[nuc_737_coords["T"]==t, ["X", "Y", "Z"]].to_numpy().flatten()
    pos_tm1=nuc_737_coords.loc[nuc_737_coords["T"]==t-1, ["X", "Y", "Z"]].to_numpy().flatten()
    pos_tp1=nuc_737_coords.loc[nuc_737_coords["T"]==t+1, ["X", "Y", "Z"]].to_numpy().flatten()
    dir_tm1=pos_t-pos_tm1
    dir_t=pos_tp1-pos_t
    #if t==1:
    Vec_Nuc_dirs.append(dir_tm1)
    if t==N_tp-2:
        Vec_Nuc_dirs.append(dir_t)
    
    cosine = np.dot(dir_tm1,dir_t)/(np.linalg.norm(dir_tm1)*np.linalg.norm(dir_t))
    Cosine_Nuc_dirs.append(cosine)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) #(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))-np.pi/2)/np.pi*2

moving_avg_range=2

for s in Unique_spots:
    spot_df=single_spots_df_all_tp[s]
    spot_df["X_avg"]=np.nan
    spot_df["Y_avg"]=np.nan
    spot_df["Z_avg"]=np.nan
    for t in range(N_tp):
        time_wdw=list(range(t-moving_avg_range, t+moving_avg_range+1))
        all_pos_twdw=spot_df.loc[spot_df["T"].isin(time_wdw), ["X_reg", "Y_reg", "Z_reg"]].to_numpy()#.flatten()
        pos_avg_t=np.nanmean(all_pos_twdw, axis=0)
        spot_df.loc[spot_df["T"]==t, ["X_avg", "Y_avg", "Z_avg"]] = pos_avg_t
        

for i in Unique_spots:
    fig, axs=plt.subplots(2, 2, figsize=(15, 15) ) #sharex=True, sharey=True
    fig.delaxes(axs[1, 1])
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.18)
    spot_df=single_spots_df_all_tp[i]
    #axs[0,0].plot(spot_df["X_reg"], spot_df["Y_reg"])
    im_xy=plot_colourline(spot_df["X_avg"], spot_df["Y_avg"], spot_df["T"], axs[0,0])
    axs[0,0].set_xlabel("X")
    axs[0,0].set_ylabel("Y")
    #axs[0,1].plot(spot_df["Z_reg"], spot_df["Y_reg"])
    im_zy=plot_colourline(spot_df["Z_avg"], spot_df["Y_avg"], spot_df["T"], axs[0,1])
    axs[0,1].set_xlabel("Z")
    axs[0,1].set_ylabel("Y")
    #axs[1,0].plot(spot_df["X_reg"], spot_df["Z_reg"])
    im_xz=plot_colourline(spot_df["X_avg"], spot_df["Z_avg"], spot_df["T"], axs[1,0])
    axs[1,0].set_xlabel("X")
    axs[1,0].set_ylabel("Z")
    
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    
    im_xyz=plot_colourline_3D(spot_df["X_avg"], spot_df["Y_avg"], spot_df["Z_avg"], spot_df["T"], ax)
    ax.set_zlabel("Z", labelpad = 10)
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    #p21 = axs[1,0].get_position()
    p22 = ax.get_position()
    
    # shift left subplot up; also change its width and height
    #p22_new = [p22.x0, (0.05*p22.height)+p22.y0, 0.8*p21.width, 0.85*p21.height]
    #p22_new = [p22.width, p22.height, p21.x0, (0.05*p21.height)+p21.y0]
    #p22_new=p22
    p22.p1=p22.p1-0.05
    ax.set_position(p22)
    fig.savefig(path_out_im+im_prefix+"spot_"+str(i)+"_position_moving_avg_p"+str(moving_avg_range)+"_m"+str(moving_avg_range)+"_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()


for i in Unique_spots:
    fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True, subplot_kw={"projection":"3d"})
    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    spot_df=single_spots_df_all_tp[i]
    #ax.plot(spot_df["X_reg"], spot_df["Y_reg"], spot_df["Z_reg"])
    im_xyz=plot_colourline_3D(spot_df["X_avg"], spot_df["Y_avg"], spot_df["Z_avg"], spot_df["T"], ax)
    ax.set_zlabel("Z")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    fig.savefig(path_out_im+im_prefix+"spot_"+str(i)+"_position_moving_avg_p"+str(moving_avg_range)+"_m"+str(moving_avg_range)+"_vs_time_3D.png", bbox_inches='tight', dpi=300)
    plt.close()

#coords_type=""
#coords_type="_reg"
#coords_type="_avg"

all_coords_types=["", "_reg", "_avg"]

for coords_type in all_coords_types:
    
    if coords_type=="":
        coords_sub_dir="ID"+str(Cell_ID)+"_Direction_and_Movement_XYZ_vanilla/"
    elif coords_type=="_reg":
        coords_sub_dir="ID"+str(Cell_ID)+"_Direction_and_Movement_XYZ_reg/"
    elif coords_type=="_avg":
        coords_sub_dir="ID"+str(Cell_ID)+"_Direction_and_Movement_XYZ_avg/"
        
    new_path_out_im=Path(path_out_im+coords_sub_dir)
    new_path_out_im.mkdir(parents=True, exist_ok=True)
    
    for i in Unique_spots:
        fig, axs=plt.subplots(2, 2, figsize=(15, 15) ) #sharex=True, sharey=True
        fig.delaxes(axs[1, 1])
        fig.subplots_adjust(hspace=0.15)
        fig.subplots_adjust(wspace=0.18)
        spot_df=single_spots_df_all_tp[i]
        #axs[0,0].plot(spot_df["X_reg"], spot_df["Y_reg"])
        im_xy=plot_colourline(spot_df["X"+coords_type], spot_df["Y"+coords_type], spot_df["T"], axs[0,0])
        axs[0,0].set_xlabel("X")
        axs[0,0].set_ylabel("Y")
        #axs[0,1].plot(spot_df["Z_reg"], spot_df["Y_reg"])
        im_zy=plot_colourline(spot_df["Z"+coords_type], spot_df["Y"+coords_type], spot_df["T"], axs[0,1])
        axs[0,1].set_xlabel("Z")
        axs[0,1].set_ylabel("Y")
        #axs[1,0].plot(spot_df["X_reg"], spot_df["Z_reg"])
        im_xz=plot_colourline(spot_df["X"+coords_type], spot_df["Z"+coords_type], spot_df["T"], axs[1,0])
        axs[1,0].set_xlabel("X")
        axs[1,0].set_ylabel("Z")
        
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        
        
        im_xyz=plot_colourline_3D(spot_df["X"+coords_type], spot_df["Y"+coords_type], spot_df["Z"+coords_type], spot_df["T"], ax)
        ax.set_zlabel("Z", labelpad = 10)
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        p22 = ax.get_position()
    
        p22.p1=p22.p1-0.05
        ax.set_position(p22)
        fig.savefig(path_out_im+coords_sub_dir+im_prefix+"spot_"+str(i)+"_position_Diff_coords_vs_time.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    angular_dist_dir_dict={}
    cosine_dir_dict={}
    all_dirs_vecs_dict={}
    norm_dirs_vecs_dict={}
    for s in Unique_spots:
       cosine_dir_dict[s]=[] 
       all_dirs_vecs_dict[s]=[]
       angular_dist_dir_dict[s]=[]
       norm_dirs_vecs_dict[s]=[]
       
    
    for s in Unique_spots:
        spot_df=single_spots_df_all_tp[s]
        for t in range(1, N_tp-1):
            #pos_t=spot_df.loc[spot_df["T"]==t, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
            #pos_tm1=spot_df.loc[spot_df["T"]==t-1, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
            #pos_tp1=spot_df.loc[spot_df["T"]==t+1, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
            
            #pos_t=spot_df.loc[spot_df["T"]==t, ["X_avg", "Y_avg", "Z_avg"]].to_numpy().flatten()
            #pos_tm1=spot_df.loc[spot_df["T"]==t-1, ["X_avg", "Y_avg", "Z_avg"]].to_numpy().flatten()
            #pos_tp1=spot_df.loc[spot_df["T"]==t+1, ["X_avg", "Y_avg", "Z_avg"]].to_numpy().flatten()
            
            #pos_t=spot_df.loc[spot_df["T"]==t, ["X", "Y", "Z"]].to_numpy().flatten()
            #pos_tm1=spot_df.loc[spot_df["T"]==t-1, ["X", "Y", "Z"]].to_numpy().flatten()
            #pos_tp1=spot_df.loc[spot_df["T"]==t+1, ["X", "Y", "Z"]].to_numpy().flatten()
            
            pos_t=spot_df.loc[spot_df["T"]==t, ["X"+coords_type, "Y"+coords_type, "Z"+coords_type]].to_numpy().flatten()
            pos_tm1=spot_df.loc[spot_df["T"]==t-1, ["X"+coords_type, "Y"+coords_type, "Z"+coords_type]].to_numpy().flatten()
            pos_tp1=spot_df.loc[spot_df["T"]==t+1, ["X"+coords_type, "Y"+coords_type, "Z"+coords_type]].to_numpy().flatten()
            
            dir_tm1=pos_t-pos_tm1
            #dir_tm1-=Vec_Nuc_dirs[t-1]
            dir_t=pos_tp1-pos_t
            #dir_t-=Vec_Nuc_dirs[t]
            #if t==1:
            all_dirs_vecs_dict[s].append(dir_tm1)
            norm_dirs_vecs_dict[s].append(np.linalg.norm(dir_tm1))
            if t==N_tp-2:
                all_dirs_vecs_dict[s].append(dir_t)
                norm_dirs_vecs_dict[s].append(np.linalg.norm(dir_t))
            
            cosine = np.dot(dir_tm1,dir_t)/(np.linalg.norm(dir_tm1)*np.linalg.norm(dir_t))
            cosine_dir_dict[s].append(cosine)
            #angular_dist_dir_dict[s].append(angle_between(dir_tm1, dir_t))
            angular_dist_dir_dict[s].append(np.arccos(cosine)/np.pi)
    """     
    all_covs=[]
    for t in range(N_tp-1):
        coord_mat_T=np.zeros((3, len(Unique_spots)))
        for s in Unique_spots:
            coord_mat_T[:,s-1]= all_dirs_vecs_dict[s][t]
        all_covs.append(np.cov(coord_mat_T))
    """     
    
    """
    for s in Unique_spots:
        spot_df=single_spots_df_all_tp[s]
        for t in range(5, 345):
            pos_t=spot_df.loc[spot_df["T"]==t, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
            pos_tm1=spot_df.loc[spot_df["T"]==t-5, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
            pos_tp1=spot_df.loc[spot_df["T"]==t+5, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
            dir_tm1=pos_t-pos_tm1
            dir_t=pos_tp1-pos_t
            cosine = np.dot(dir_tm1,dir_t)/(np.linalg.norm(dir_tm1)*np.linalg.norm(dir_t))
            cosine_dir_dict[s].append(cosine)
    """
    
    fig, axs=make_fig_2(share_x=False, share_y=False)
    for i, ax in zip(Unique_spots, axs.flatten()):
        cosine_spot=cosine_dir_dict[i]
        ax.plot(range(1,N_tp-1), cosine_spot)
        ax.set_ylabel("Cosine similarity between direction vectors (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_cosine_similarity_direction_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    """
    fig, ax=plt.subplots(1, 1, figsize=(24, 12), sharex=True, sharey=True)   
    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    for i in Unique_spots:
        cosine_spot=cosine_dir_dict[i]
        ax.plot(range(1,N_tp-1), cosine_spot)
        ax.set_ylabel("Cosine similarity between direction vectors (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+"mean_fluo_vs_time_id"+str(Cell_ID)+"/"+im_prefix+"single_cosine_similarity_direction_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    """
    p1_t_count={}
    b1_t_count={}
    for i in Unique_spots:
        cosine_spot=cosine_dir_dict[i]
        c=1
        #count_sim=[]
        for sim in cosine_spot:
            if sim>=0.7:
                if c in p1_t_count.keys():
                    p1_t_count[c]+=1
                else:
                    p1_t_count[c]=1
                c+=1
            else:
                if c in b1_t_count.keys():
                    b1_t_count[c]+=1
                else:
                    b1_t_count[c]=1
                c=1
    
    
    p1_t_prob={}
    for l, p1_t_N in p1_t_count.items():
        try:
            b1_t_N=b1_t_count[l]
        except KeyError:
            b1_t_N=0
        
        p1_t_prob[l]=p1_t_N/(p1_t_N+b1_t_N)
        
    
    
    
    fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True)
    ax.bar(np.array(list(p1_t_prob.keys()))-1, list(p1_t_prob.values()))
    ax.set_ylabel("Prob to stay in similar direction")
    ax.set_xlabel("Successive time steps without direction change")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Transition_probabilities_similarity_direction_all_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    
    rdm.seed(42)
    max_length=max(p1_t_prob.keys())
    all_sim_version=["Uniform", "T_prob"]
    for sim_version in all_sim_version:
        #sim_version="T_prob" #"Uniform" "T_prob"
        #sim_version="Uniform"
        
        
        cosine_dir_simulation_dict={}
        for s in Unique_spots:
            cosine_dir_simulation_dict[s]=[]
            c=1
            for i in range(N_tp-2):
                if sim_version=="Uniform":
                    rdm_angle=rdm.uniform(0, 2*np.pi)
                elif sim_version=="T_prob":
                    same_dir=["Yes", "No"]
                    if c==max_length+1:
                        rdm_angle=rdm.uniform( np.arccos(0.7), 2*np.pi-np.arccos(0.69))
                        c=1
                    else:
                        prob_same_dir=p1_t_prob[c]
                        choice_same_dir=rdm.choices(same_dir, weights=[prob_same_dir, 1-prob_same_dir])[0]
                        if choice_same_dir=="Yes":
                            rdm_angle=rdm.uniform(0, np.arccos(0.7))
                            c+=1
                        else:
                            rdm_angle=rdm.uniform( np.arccos(0.7), 2*np.pi-np.arccos(0.69))
                            c=1
                 
                cosine_dir_simulation_dict[s].append(np.cos(rdm_angle))
               
        fig, axs=plt.subplots(N_H_plots_2, N_W_plots_2, figsize=(fig_width_2, fig_height_2)) #, sharex=True, sharey=True
            
        fig.subplots_adjust(hspace=0.075)
        fig.subplots_adjust(wspace=0.075)
        for i, ax in zip(Unique_spots, axs.flatten()):
            cosine_spot=cosine_dir_simulation_dict[i]
            ax.plot(range(1,N_tp-1), cosine_spot)
            ax.set_ylabel("Cosine similarity between direction vectors (spot "+str(i)+")")
            ax.set_xlabel("Time frame (a.u)")
        fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_cosine_similarity_direction_random_"+sim_version+"_simulation_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        
        p1_t_count_rdm={}
        b1_t_count_rdm={}
        for i in Unique_spots:
            cosine_spot=cosine_dir_simulation_dict[i]
            c=1
            #count_sim=[]
            for sim in cosine_spot:
                if sim>=0.7:
                    if c in p1_t_count_rdm.keys():
                        p1_t_count_rdm[c]+=1
                    else:
                        p1_t_count_rdm[c]=1
                    c+=1
                else:
                    if c in b1_t_count_rdm.keys():
                        b1_t_count_rdm[c]+=1
                    else:
                        b1_t_count_rdm[c]=1
                    c=1
        
        
        p1_t_prob_rdm={}
        for l, p1_t_N in p1_t_count_rdm.items():
            try:
                b1_t_N=b1_t_count_rdm[l]
            except KeyError:
                b1_t_N=0
            
            p1_t_prob_rdm[l]=p1_t_N/(p1_t_N+b1_t_N)
            
        
        
        
        fig, ax=plt.subplots(figsize=(14, 7), sharex=True, sharey=True)
        ax.bar(np.array(list(p1_t_prob_rdm.keys()))-1, list(p1_t_prob_rdm.values()))
        ax.set_ylabel("Prob to stay in similar direction")
        ax.set_xlabel("Successive time steps without direction change")
        fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Transition_probabilities_similarity_direction_all_random_"+sim_version+"_simulation_vs_time.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    
    
    
    fig, axs=make_fig_2(share_x=False, share_y=False)
    for i, ax in zip(Unique_spots, axs.flatten()):
        cosine_spot=cosine_dir_dict[i]
        c=1
        count_sim=[]
        for sim in cosine_spot:
            if sim>=0.7:
                c+=1
            else:
                count_sim.append(c)
                c=1
            
        ax.hist(count_sim, bins=max(count_sim)-1, density=True) # 
        ax.set_ylabel("freq")
        ax.set_xlabel("lenght of simcos (spot "+str(i)+")")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_cosine_similarity_direction_length_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    
    
    
    
    
    fig, axs=make_fig_2()
    for i, ax in zip(Unique_spots, axs.flatten()):
        angle_spot=angular_dist_dir_dict[i]
        ax.plot(range(1,N_tp-1), angle_spot)
        ax.set_ylabel("Angular distance between direction vectors (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_angular_distance_direction_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    
    
    
    fig, axs=make_fig_2()
    for i, ax in zip(Unique_spots, axs.flatten()):
        cosine_spot=np.array(cosine_dir_dict[i])
        ax.plot(range(1,N_tp-2), cosine_spot[1:]-cosine_spot[:-1])
        ax.set_ylabel("Cosine similarity between direction vectors (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_cosine_similarity_direction_derivative_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, axs=make_fig_1()
    for i, ax in zip(Unique_spots, axs.flatten()):
        cosine_spot=np.array(cosine_dir_dict[i])
        d1=cosine_spot[1:]-cosine_spot[:-1]
        ax.plot(range(1,N_tp-3), d1[1:]-d1[:-1])
        ax.set_ylabel("Cosine similarity between direction vectors (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_cosine_similarity_direction_derivative2_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    spots_X_to_df=np.zeros((len(Unique_spots), N_tp))
    spots_Y_to_df=np.zeros((len(Unique_spots), N_tp))
    spots_Z_to_df=np.zeros((len(Unique_spots), N_tp))
    for s in Unique_spots:
        spot_df=single_spots_df_all_tp[s]
        spots_X_to_df[s-1]=spot_df["X"+coords_type].to_numpy()
        spots_Y_to_df[s-1]=spot_df["Y"+coords_type].to_numpy()
        spots_Z_to_df[s-1]=spot_df["Z"+coords_type].to_numpy()
    
    corr_coords_method="pearson"
    spots_X_df=pd.DataFrame(spots_X_to_df)
    spots_Y_df=pd.DataFrame(spots_Y_to_df)
    spots_Z_df=pd.DataFrame(spots_Z_to_df)
    corr_mat_X=spots_X_df.T.corr(method=corr_coords_method).to_numpy()
    corr_mat_Y=spots_Y_df.T.corr(method=corr_coords_method).to_numpy()
    corr_mat_Z=spots_Z_df.T.corr(method=corr_coords_method).to_numpy()
    corr_mat_coords_mean=(corr_mat_X+corr_mat_Y+corr_mat_Z)/3
    
    fig, ax=plt.subplots() #figsize=(10,10)
    coord_plot=ax.imshow(corr_mat_X, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
    ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
    ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
    #ax.set_xlabel("Theta bin")
    #ax.set_ylabel("Phi bin")
    for i in range(len(Unique_spots)):
        for j in range(len(Unique_spots)):
            cell_value = corr_mat_X[i, j]
            cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
            
            # Determine whether the cell color is dark or light
            #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
            if luminance > 0.5:
                text_color = 'black'  # Use black for light backgrounds
            else:
                text_color = 'white'  # Use white for dark backgrounds
            
            ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                         xy=(j, i),
                         ha='center', va='center',
                         fontsize=7.5,
                         color=text_color)
    fig.colorbar(coord_plot, ax=ax, label=corr_coords_method+" correlation coefficient")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Time_"+corr_coords_method+"_correlation_X_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
    plt.close()
    
    
    fig, ax=plt.subplots() #figsize=(10,10)
    coord_plot=ax.imshow(corr_mat_Y, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
    ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
    ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
    #ax.set_xlabel("Theta bin")
    #ax.set_ylabel("Phi bin")
    for i in range(len(Unique_spots)):
        for j in range(len(Unique_spots)):
            cell_value = corr_mat_Y[i, j]
            cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
            
            # Determine whether the cell color is dark or light
            #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
            if luminance > 0.5:
                text_color = 'black'  # Use black for light backgrounds
            else:
                text_color = 'white'  # Use white for dark backgrounds
            
            ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                         xy=(j, i),
                         ha='center', va='center',
                         fontsize=7.5,
                         color=text_color)
    fig.colorbar(coord_plot, ax=ax, label=corr_coords_method+" correlation coefficient")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Time_"+corr_coords_method+"_correlation_Y_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
    plt.close()
    
    fig, ax=plt.subplots() #figsize=(10,10)
    coord_plot=ax.imshow(corr_mat_Z, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
    ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
    ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
    #ax.set_xlabel("Theta bin")
    #ax.set_ylabel("Phi bin")
    for i in range(len(Unique_spots)):
        for j in range(len(Unique_spots)):
            cell_value = corr_mat_Z[i, j]
            cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
            
            # Determine whether the cell color is dark or light
            #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
            if luminance > 0.5:
                text_color = 'black'  # Use black for light backgrounds
            else:
                text_color = 'white'  # Use white for dark backgrounds
            
            ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                         xy=(j, i),
                         ha='center', va='center',
                         fontsize=7.5,
                         color=text_color)
    fig.colorbar(coord_plot, ax=ax, label=corr_coords_method+" correlation coefficient")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Time_"+corr_coords_method+"_correlation_Z_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
    plt.close()
    
    fig, ax=plt.subplots() #figsize=(10,10)
    coord_plot=ax.imshow(corr_mat_coords_mean, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
    ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
    ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
    #ax.set_xlabel("Theta bin")
    #ax.set_ylabel("Phi bin")
    for i in range(len(Unique_spots)):
        for j in range(len(Unique_spots)):
            cell_value = corr_mat_coords_mean[i, j]
            cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
            
            # Determine whether the cell color is dark or light
            #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
            if luminance > 0.5:
                text_color = 'black'  # Use black for light backgrounds
            else:
                text_color = 'white'  # Use white for dark backgrounds
            
            ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                         xy=(j, i),
                         ha='center', va='center',
                         fontsize=7.5,
                         color=text_color)
    fig.colorbar(coord_plot, ax=ax, label=corr_coords_method+" correlation coefficient")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Time_"+corr_coords_method+"_correlation_coords_mean_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
    plt.close()
    
    spots_dir_angle_to_df=np.zeros((len(Unique_spots), N_tp-2))
    spots_dir_norm_to_df=np.zeros((len(Unique_spots), N_tp-2))
    for s in Unique_spots:
        spots_dir_angle_to_df[s-1]=angular_dist_dir_dict[s]
        spots_dir_norm_to_df[s-1]=norm_dirs_vecs_dict[s][1:]
    
    corr_vec_method="pearson"
    spots_dir_angle_df=pd.DataFrame(spots_dir_angle_to_df)
    spots_dir_norm_df=pd.DataFrame(spots_dir_norm_to_df)
    corr_mat_dir_angle=spots_dir_angle_df.T.corr(method=corr_vec_method).to_numpy()
    corr_mat_dir_norm=spots_dir_norm_df.T.corr(method=corr_vec_method).to_numpy()
    
    
    fig, ax=plt.subplots() #figsize=(10,10)
    coord_plot=ax.imshow(corr_mat_dir_angle, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
    ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
    ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
    #ax.set_xlabel("Theta bin")
    #ax.set_ylabel("Phi bin")
    for i in range(len(Unique_spots)):
        for j in range(len(Unique_spots)):
            cell_value = corr_mat_dir_angle[i, j]
            cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
            
            # Determine whether the cell color is dark or light
            #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            luminance = 0.2 * cell_color[0] + 0.2 * cell_color[1] + 0.6 * cell_color[2]
            if luminance > 0.5:
                text_color = 'black'  # Use black for light backgrounds
            else:
                text_color = 'white'  # Use white for dark backgrounds
            
            ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                         xy=(j, i),
                         ha='center', va='center',
                         fontsize=7.5,
                         color=text_color)
    fig.colorbar(coord_plot, ax=ax, label=corr_vec_method+" correlation coefficient")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Time_"+corr_vec_method+"_correlation_dir_angle_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
    plt.close()
    
    fig, ax=plt.subplots() #figsize=(10,10)
    coord_plot=ax.imshow(corr_mat_dir_norm, aspect="auto", cmap="bwr", vmin=-1, vmax=1) #, vmax=0.3
    ax.set_xticks(np.array(Unique_spots)-1, Unique_spots)
    ax.set_yticks(np.array(Unique_spots)-1, Unique_spots)
    #ax.set_xlabel("Theta bin")
    #ax.set_ylabel("Phi bin")
    for i in range(len(Unique_spots)):
        for j in range(len(Unique_spots)):
            cell_value = corr_mat_dir_norm[i, j]
            cell_color = plt.cm.bwr(cell_value)  # Get the color of the cell
            
            # Determine whether the cell color is dark or light
            #luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
            luminance = 0.2 * cell_color[0] + 0.22 * cell_color[1] + 0.6 * cell_color[2]
            if luminance > 0.5:
                text_color = 'black'  # Use black for light backgrounds
            else:
                text_color = 'white'  # Use white for dark backgrounds
            
            ax.annotate(f'{round(cell_value, 2)}', #f'{cell_value:.2f}'
                         xy=(j, i),
                         ha='center', va='center',
                         fontsize=7.5,
                         color=text_color)
    fig.colorbar(coord_plot, ax=ax, label=corr_vec_method+" correlation coefficient")
    ax.set_box_aspect(1)
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Time_"+corr_vec_method+"_correlation_dir_norm_single_spots.png", dpi=300, bbox_inches='tight') #bbox_inches='tight',
    plt.close()
    
    
    
    fig, axs=make_fig_2()
    for i, ax in zip(Unique_spots, axs.flatten()):
        ax.plot(range(1,N_tp-1), spots_dir_norm_to_df[i-1])
        ax.set_ylabel("Norm of direction vector (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"single_direction_vector_norm_per_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax=plt.subplots(figsize=(24, 12), sharex=True, sharey=True)
        
    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    for i in Unique_spots:
        ax.plot(range(1,N_tp-1), spots_dir_norm_to_df[i-1])
        ax.set_ylabel("Norm of direction vector (spot "+str(i)+")")
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+coords_sub_dir+im_prefix+"Direction_vector_norm_all_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()




spots_comb=list(combinations(Unique_spots, 2))
N_comb=len(spots_comb)

dict_dist_between_spots={}
for comb in spots_comb:
    dict_dist_between_spots[comb]=[]

for comb in spots_comb:
    spot_1=comb[0]
    spot_2=comb[1]
    spot_df_1=single_spots_df_all_tp[spot_1]
    spot_df_2=single_spots_df_all_tp[spot_2]
    for t in range(N_tp):
        pos_1=spot_df_1.loc[spot_df_1["T"]==t, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
        pos_2=spot_df_2.loc[spot_df_2["T"]==t, ["X_reg", "Y_reg", "Z_reg"]].to_numpy().flatten()
        dist=np.sqrt(np.sum((pos_2-pos_1)**2))
        dict_dist_between_spots[comb].append(dist)

if Cell_ID==135 or Cell_ID==10:
    fig, ax=plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)
    c=spots_comb[0]
    dist_comb=dict_dist_between_spots[c]
    spot_1=c[0]
    spot_2=c[1]
    ax.plot(range(N_tp), dist_comb)
    ax.set_ylabel("Distance between spots "+str(spot_1)+" and "+str(spot_2))
    ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+im_prefix+"single_dist_btw_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()
    
elif Cell_ID==737:
    fig, axs=plt.subplots(5, 3, figsize=(36, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    for c, ax in zip(spots_comb, axs.flatten()):
        dist_comb=dict_dist_between_spots[c]
        spot_1=c[0]
        spot_2=c[1]
        ax.plot(range(N_tp), dist_comb)
        ax.set_ylabel("spots "+str(spot_1)+" and "+str(spot_2))
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+im_prefix+"single_dist_btw_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()

elif Cell_ID==540:
    fig, axs=plt.subplots(11, 6, figsize=(72, 26), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.075)
    fig.subplots_adjust(wspace=0.075)
    for c, ax in zip(spots_comb, axs.flatten()):
        dist_comb=dict_dist_between_spots[c]
        spot_1=c[0]
        spot_2=c[1]
        ax.plot(range(N_tp), dist_comb)
        ax.set_ylabel("spots "+str(spot_1)+" and "+str(spot_2))
        ax.set_xlabel("Time frame (a.u)")
    fig.savefig(path_out_im+im_prefix+"single_dist_btw_spots_vs_time.png", bbox_inches='tight', dpi=300)
    plt.close()



    #angular_dist_dir_dict[s].append(angle_between(dir_tm1, dir_t))

fig, ax=plt.subplots(1, 1, figsize=(24, 12), sharex=True, sharey=True)

    
fig.subplots_adjust(hspace=0.075)
fig.subplots_adjust(wspace=0.075)
ax.plot(range(1,N_tp-1), Cosine_Nuc_dirs)
ax.set_ylabel("Cosine similarity between direction vectors of Nuc737")
ax.set_xlabel("Time frame (a.u)")
fig.savefig(path_out_im+im_prefix+"single_cosine_similarity_direction_Nuc_vs_time.png", bbox_inches='tight', dpi=300)
plt.close()





if Open_napari:
    
    viewer = napari.Viewer(ndisplay=2)
    
    
    image_layer = viewer.add_image(stack_C1, scale=scale, colormap='green', blending='additive',visible=True, contrast_limits=[90, 400])
    
    #image_layer = viewer.add_image(stack_C1, scale=scale, colormap='green', blending='additive',visible=True)
    
    image_layer = viewer.add_image(stack_C2, scale=scale, colormap='magenta', blending='additive',visible=True, contrast_limits=[90, 500])
    
    #image_layer = viewer.add_labels(stack_C3, scale=scale, blending='additive',visible=True)
    viewer.add_tracks(data, properties=properties, graph=graph, visible=True, colormap="turbo", blending="additive") # scale=scale
    
    viewer.add_tracks(data_spots, properties=properties_spots, graph=graph_spots, visible=True, colormap="turbo", blending="additive") # scale=scale
    
    points_layer = viewer.add_points(spots_tzyx_nuc, ndim=4, size=200, scale=scale, blending='additive', opacity=0.3) #ndim=4
    
    #if Cell_ID==737:
    points_layer = viewer.add_points(Corner_spots, ndim=4, size=200, scale=scale, blending='additive', opacity=0.3, face_color="yellow") #ndim=4
    #mask=points_layer.to_masks(stack_C1.shape)
    #viewer.camera.angles = (-0.26571224801734533, -3.2N_tp-1084850881065, 146.03256463889608)
    
    
    
    viewer.camera.zoom=13.479914708057303
    
    viewer.dims.current_step = (0 , 24, 540, 430)
    
    if Record_Movie:

        def center_camera_on_object():
            #ids_nuc_to_track=(135, 5)
            #ids_nuc_to_track=(27, 4) #Non-muscle_cell
            #ids_nuc_to_track=[10]
            tp = viewer.dims.current_step[0]
            """
            if tp<81 or 87<tp<92: 
                ids_nuc_to_track=(737, 312, 96, 415, 449, 336, 7)
            elif tp<113:
                ids_nuc_to_track=(737, 415, 449, 336, 312, 96, 7)
            else:
                ids_nuc_to_track=(737, 449, 415, 336, 312, 96, 7)
            """
            #ids_nuc_to_track=tuple([10])
            
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
                
            viewer.camera.center=(coords.Z.values[0], coords.Y.values[0], coords.X.values[0])
            viewer.dims.current_step = (tp , coords.Z.values[0], coords.Y.values[0], coords.X.values[0])
            
        
        #viewer.dims.events.current_step.connect(center_camera_on_object)
        
        animation = Animation(viewer)
        #viewer.camera.angles=(0.1597470119177224, -3.358462402302114, 140.39807369447243)
        
        viewer.dims.current_step = (0 , 24, 540, 430)
        animation.capture_keyframe(steps=10)
        
        viewer.dims.events.current_step.connect(center_camera_on_object)
        viewer.dims.current_step = (N_tp-1, 24, 540, 430)
        animation.capture_keyframe(steps=N_tp-1)
        
        #animation.capture_keyframe(steps=60)
        
        
        #Please write path and filenmae in which to output the animation
        #animation.animate("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Centrioles_particle_tracking_with_tracks_only_id"+str(Cell_ID)+"_topview.mp4", canvas_only=True)
        animation.animate(path_out_movies+"Centrioles_and_nuc_particle_tracking_with_tracks_id"+str(Cell_ID)+"_topview.mp4", canvas_only=True)





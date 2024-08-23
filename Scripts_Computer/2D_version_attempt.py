#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:00:54 2024

@author: floriancurvaia
"""


from tifffile import imread
import numpy as np
from pathlib import Path
import pandas as pd
import napari 
import btrack

path_folder=Path("F:/GG/Live/Napari_tracks/").as_posix()


path_to_image=Path("F:/GG/Live/Napari_tracks/MAX_20240709_ZM_RPE1p53KO_Cep63+mScar_Cep135+GFP_CS650_200nM_9hCen...+mScar_Cep135+GFP_CS650_200nM_9hCentWO_1_MMStack_1-Pos_001_000.tif").as_posix()


path_to_csv=Path("F:/GG/Live/Napari_tracks/MAX_20240709_ZM_RPE1p53KO_Cep63+mScar_Cep135+GFP_CS650_200nM_9hCen...+mScar_Cep135+GFP_CS650_200nM_9hCentWO_1_MMStack_1-Pos_001_000_allspots.csv").as_posix()

Number_of_dim=2

Time_interval=600

Number_of_tp=106

spots_df=pd.read_csv(path_to_csv, sep=",")
spots_df.drop([0,1,2], axis=0, inplace=True)
spots_df[['POSITION_T', 'POSITION_X', 'POSITION_Y', 'POSITION_Z']]=spots_df[['POSITION_T', 'POSITION_X', 'POSITION_Y', 'POSITION_Z']].astype('float64')
spots_df["POSITION_T"]=spots_df["POSITION_T"]/Time_interval
if Number_of_dim==2:
    spots_to_napari=spots_df[["POSITION_T", "POSITION_Y", "POSITION_X"]].to_numpy()
    
if Number_of_dim==3:
    spots_to_napari=spots_df[["POSITION_T", "POSITION_Z", "POSITION_Y", "POSITION_X"]].to_numpy()

Number_of_channels=3

image = imread(path_to_image)

scale=(0.7, 0.2050, 0.2050) #There are pixel size in Z, Y, X
scale=(1, 0.2050, 0.2050) #There are pixel size in Z, Y, X

colormaps=["green", "magenta", "cyan"]


#viewer = napari.Viewer(ndisplay=2)
for i in range(Number_of_channels):
    image_layer = viewer.add_image(image[:, i, :, :],  name="Channel "+str(i), scale=scale, colormap=colormaps[i], blending='additive',visible=True)

points_layer = viewer.add_points(spots_to_napari, size=5, blending='additive',  opacity=0.3) #ndim=4 ndim=Number_of_dim+1, 


######CURATE SPOTS########



cur_spots_to_349=viewer.layers["spots_to_napari"].data.copy()
cur_spots_to_349=cur_spots_to_349[np.isnan(cur_spots_to_349.sum(1))==False]
cur_spots_to_349[:, [1,2]]=cur_spots_to_349[:, [2,1]]
cur_spots_349=pd.DataFrame(cur_spots_to_349, columns=["T", "Y", "X"])
cur_spots_349.to_csv(path_folder+"spots_to_napari.csv")



path_config=Path(path_folder)

new_nuc_objs=btrack.io.localizations_to_objects(localizations=cur_spots_to_349) #/scale_w_T 

with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    #tracker.configure(path_config /'cell_config.json')
    
    tracker.configure(path_config /'particle_config_2.json')
    
    # append the objects to be tracked
    tracker.append(new_nuc_objs)
    
    # set the volume (Z axis volume limits default to [-1e5, 1e5] for 2D data)
    tracker.volume = ( (0, 250), (0, 175))
    
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
    
    tracker.export(path_folder+'FOV1_CentWO.h5', obj_type='obj_type_1')
  
#data_spots[:, 0]=data_spots[:,0].as_type(int)
viewer.add_tracks(data_spots, properties=properties_spots, graph=graph_spots, visible=True, colormap="turbo", blending="additive")
path_files=Path(path_folder)

np.save(path_files / "spots_coords_particle_track.npy", data_spots)


  
###CURATE TRACKING####


dict_all_tracks={}
N_tracks=6
last_tp_tracks=[6, 16, 18, 19, 21, 32]

dict_all_tracks_w_merge={}

for i in range(1, N_tracks+1):
    dict_all_tracks[i]={}
    dict_all_tracks_w_merge[i]={}

#last_tp_tracks=np.array([6, 16, 18, 19, 21, 32])

test=[]
with open(path_files / "Spots_time_ID_toDict.txt" ) as f:
    lines=f.readlines()
    tp_ID=None
    upper_tp=None
    track_ID=None
    tp_ID_merge=None
    for line in lines:
        row=line.strip()
        chunks=row.split(" ")
        line_start=chunks[0]
        if not line_start:
            continue
        if line_start == "ID":
            if tp_ID is not None:
                lower_tp=-1
                for i in range(upper_tp, lower_tp, -1):
                    dict_all_tracks[track_ID][i]=tp_ID
                    dict_all_tracks_w_merge[track_ID][i]=tp_ID_merge
                
            last_tp_ID=int(chunks[1])
            track_ID=last_tp_tracks.index(last_tp_ID) + 1
            tp_ID=last_tp_ID
            tp_ID_merge=last_tp_ID
            upper_tp=Number_of_tp
            continue
        
        lower_tp=int(line_start.split(":")[0])
        for i in range(upper_tp, lower_tp, -1):
            dict_all_tracks[track_ID][i]=tp_ID
            dict_all_tracks_w_merge[track_ID][i]=tp_ID_merge
        upper_tp=lower_tp
        if chunks[1].isdigit():
            tp_ID= int(chunks[1])
            tp_ID_merge= int(chunks[1])
            
        elif chunks[1] == "Disappear":
            if len(chunks) > 2:
                tp_ID= np.nan
                tp_ID_merge= int(chunks[-1])
            else:
                tp_ID= np.nan
                tp_ID_merge= np.nan
                
    lower_tp=-1
    for i in range(upper_tp, lower_tp, -1):
        dict_all_tracks[track_ID][i]=tp_ID
        dict_all_tracks_w_merge[track_ID][i]=tp_ID_merge
                
        #test.append(line_start)

#data_new=np.load(path_files / "spots_coords_particle_track.npy")
spots_track_coords=pd.DataFrame(data_spots, columns=["ID", "T", "Y", "X"])
spots_track_coords.ID=spots_track_coords.ID+N_tracks
scale=(0.75, 0.173, 0.173)

"""
spots_tzyx_nuc_df=pd.read_csv(path_files / "spots_zhansaya.csv")
spots_tzyx_nuc_df[["Y", "X"]]=spots_tzyx_nuc_df[[ "Y", "X"]]*scale
spots_tzyx_nuc_df.sort_values("T", inplace=True)
"""
#t1=[]
#t2=[]
#spots_track_coords.loc[len(spots_track_coords)] = np.append(3, cur_spots_to_349[-1]*new_scale)
all_spots=[]
for track_ID, tp_dict in dict_all_tracks.items():
    for tp, tp_ID in tp_dict.items():
        if not np.isnan(tp_ID):
            spot=spots_track_coords.loc[(spots_track_coords["ID"]==tp_ID+N_tracks) & (spots_track_coords["T"]==tp)].copy() #, ["T", "Z", "Y", "X"]
            spot.ID=track_ID
            all_spots.append(spot)
    #t1.append(track_ID)
    #t2.append(tp_dict[0])

cur_spots=pd.concat(all_spots, axis=0, ignore_index=True)
cur_spots.sort_values(["T", "ID"], inplace=True)
cur_spots.to_csv(path_files / "curated_track_spots_zhansaya.csv")

data_new=spots_track_coords[["ID","T", "Y", "X"]].to_numpy()
np.save(path_files / "curated_spots_coords_particle_track.npy", data_new)
graph_new={}
properties_new={}
viewer.add_tracks(data_new, properties=properties_new, graph=graph_new, visible=True, colormap="turbo", blending="additive")


###Measure and plot Intensity###

path_config=path_files

spots_track_coords=pd.read_csv(path_config / "curated_track_spots_zhansaya.csv")

h, w = image.shape[2:]  #100, 100, 10
colnums, rownums = np.meshgrid(range(h), range(w), indexing='ij')
r=2.5


for c in range(Number_of_channels):
    spots_track_coords["Chan_"+str(c)]=np.nan
    for tp in range(Number_of_tp):
        channel=image[tp, c, :, :]
        spots_tp=spots_track_coords.loc[spots_track_coords["T"]==tp]
        
        for ind, p in spots_tp.iterrows():
            cy, cx= (p[["Y", "X"]].to_numpy() /scale[1:]).flatten().astype(int)
            dist = np.sqrt(
                    ((colnums.flatten() - cy)*0.75) ** 2 + 
                    ((rownums.flatten() - cx)*0.75) ** 2 
                    )
            #spots_track_coords_135.iloc[ind][["Nuc_ID", "Dist_nuc"]]=[loc_ind, dist_p]
            keep_mask = (dist < r).reshape((h, w))
            indices = np.where(keep_mask)
            values=channel[indices]
            mean_values=np.mean(values)
            spots_track_coords.loc[(spots_track_coords["T"]==tp) & (spots_track_coords["ID"]==p.ID), "Chan_"+str(c)]=mean_values
            
            

            
spots_track_coords.to_csv(path_config / "curated_track_spots_zhansaya_w_fluo.csv")
        
path_out_im=path_folder

channel_names=["CenSpark650", "Cep63-mScarlet", "Cep135-GFP"]

for i in np.unique(spots_track_coords.ID):
    spots_df=spots_track_coords.loc[spots_track_coords["ID"]==i]
    fig, ax=plt.subplots(figsize=(10, 5))
    for c in range(Number_of_channels):
        ax.plot(spots_df["T"], spots_df["Chan_"+str(c)], label=channel_names[c]) #"Chan_"+str(c)
    ax.set_ylabel("Mean fluo intensity per spot")
    ax.set_xlabel("Time frame (a.u)")
    plt.legend()
    fig.savefig(path_out_im+"single_Mean_fluo_per_spots_vs_time_ID_"+str(int(i))+".png", bbox_inches='tight', dpi=300)
    plt.close()
    


  
  
  
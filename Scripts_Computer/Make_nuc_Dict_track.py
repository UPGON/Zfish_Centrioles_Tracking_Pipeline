#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:07:24 2024

@author: floriancurvaia
"""


from pathlib import Path
import numpy as np
import pandas as pd
from scipy import spatial
import pickle 

N_tp=350

Cell_ID=540
path_files=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing")

dict_all_tracks={}
N_tracks=2
last_tp_tracks=[6, 8]
#last_tp_tracks=np.array([6, 16, 18, 19, 21, 32])
dict_all_tracks_w_merge={}

for i in range(1, N_tracks+1):
    dict_all_tracks[i]={}
    dict_all_tracks_w_merge[i]={}



with open(path_files / "Nuc_time_ID_id"+str(Cell_ID)+"_toDict.txt" ) as f:
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
            upper_tp=349
            lower_tp=0
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
        
        elif chunks[1] == "nan":
            tp_ID= np.nan
            tp_ID_merge= np.nan
                
    lower_tp=-1
    for i in range(upper_tp, lower_tp, -1):
        dict_all_tracks[track_ID][i]=tp_ID
        dict_all_tracks_w_merge[track_ID][i]=tp_ID_merge
                
        #test.append(line_start)


nuc_coords_or_df=pd.read_csv(path_files/"nuc_coords_id"+str(Cell_ID)+".csv")
nuc_coords_or_df.drop("Unnamed: 0", axis=1, inplace=True)
new_nuc_track_coords=pd.read_csv(path_files/"new_all_nuc_coords_tracks_id"+str(Cell_ID)+".csv")
new_nuc_track_coords.drop("Unnamed: 0", axis=1, inplace=True)
Cell_CM_tp_dict={}
nucs_ids_per_tp={}
for i in range(N_tp):
    Cell_CM_tp_arr=np.zeros((N_tracks, 3))
    Cell_CM_tp_arr[:]=np.nan
    Cell_CM_tp_dict[i]=Cell_CM_tp_arr
    nucs_ids_per_tp[i]=[]

for track_ID, tp_dict in dict_all_tracks.items():
    for tp, tp_ID in tp_dict.items():
        if not np.isnan(tp_ID):
            nuc_coords=new_nuc_track_coords.loc[(new_nuc_track_coords["ID"]==tp_ID) & (new_nuc_track_coords["T"]==tp), ["X", "Y", "Z"]].copy()
            or_nuc_coords=nuc_coords_or_df.loc[nuc_coords_or_df["T"]==tp, ["ID", "X", "Y", "Z"]].copy()
            tree = spatial.KDTree(or_nuc_coords[["X", "Y", "Z"]].to_numpy())
            dist_p, loc_ind= tree.query(nuc_coords.to_numpy())
            or_ID=or_nuc_coords.iloc[loc_ind]["ID"].values[0]
            dict_all_tracks[track_ID][tp]=int(or_ID)
            Cell_CM_tp_dict[tp][track_ID-1]=nuc_coords
            nucs_ids_per_tp[tp].append(int(or_ID))

Cell_CM=np.zeros((N_tp, 3))
for i in range(N_tp):
    Cell_CM_tp_arr=Cell_CM_tp_dict[i]
    Cell_CM[i]=np.nanmean(Cell_CM_tp_arr, axis=0)

for i in range(N_tp):
    Cell_CM_tp=Cell_CM[i]
    fake_tp_m1=i
    fake_tp_p1=i
    while np.isnan(Cell_CM_tp.sum()):
        fake_tp_m1-=1
        fake_tp_p1+=1
        Cell_CM_m1=Cell_CM[fake_tp_m1]
        Cell_CM_p1=Cell_CM[fake_tp_p1]
        if not np.isnan(Cell_CM_m1.sum()):
            fake_tp_m1+=1
        if not np.isnan(Cell_CM_p1.sum()):
            fake_tp_p1-=1
        Cell_CM_tp=(Cell_CM_m1+Cell_CM_p1)/2
    Cell_CM[i]=Cell_CM_tp
            
            
        
np.save(path_files / 'Cell_CM_tp_id'+str(Cell_ID)+'.npy', Cell_CM)

with open(path_files/'Nuc_ID_tp_id'+str(Cell_ID)+'.pkl', 'wb+') as f:
    pickle.dump(dict_all_tracks, f)

with open(path_files/'All_nuc_IDs_tp_id'+str(Cell_ID)+'.pkl', 'wb+') as f:
    pickle.dump(nucs_ids_per_tp, f)

"""
spots_tzyx_nuc_df=pd.read_csv(path_files / "cur_spots_t349_id737_v2.csv")
spots_track_coords=pd.DataFrame(data_spots, columns=["ID", "T", "Z", "Y", "X"])
spots_track_coords.ID=spots_track_coords.ID+N_tracks
scale=(0.75, 0.173, 0.173)

spots_tzyx_nuc_df[["Z", "Y", "X"]]=spots_tzyx_nuc_df[["Z", "Y", "X"]]*scale
spots_tzyx_nuc_df.sort_values("T", inplace=True)

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
cur_spots.to_csv(path_files / "all_cur_spots_id737.csv")

all_spots=[]
for track_ID, tp_dict in dict_all_tracks_w_merge.items():
    for tp, tp_ID in tp_dict.items():
        if not np.isnan(tp_ID):
            spot=spots_track_coords.loc[(spots_track_coords["ID"]==tp_ID+N_tracks) & (spots_track_coords["T"]==tp)].copy() #, ["T", "Z", "Y", "X"]
            spot.ID=track_ID
            all_spots.append(spot)
    #t1.append(track_ID)
    #t2.append(tp_dict[0])

cur_spots=pd.concat(all_spots, axis=0, ignore_index=True)
cur_spots.sort_values(["T", "ID"], inplace=True)
cur_spots.to_csv(path_files / "all_cur_spots_id737_w_Merge.csv")
"""

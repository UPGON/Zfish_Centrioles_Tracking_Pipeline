#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:23:59 2024

@author: floriancurvaia
"""


from pathlib import Path
import numpy as np
import pandas as pd

path_files=Path("/Users/floriancurvaia/Desktop/Uni/EPFL/Gönczy/Scripts/Images/Live_transplants/Smoothing")

Cell_ID=540
dict_all_tracks={}
N_tracks=13
scale=(0.75, 0.173, 0.173)

dict_all_tracks_w_merge={}

for i in range(1, N_tracks+1):
    dict_all_tracks[i]={}
    dict_all_tracks_w_merge[i]={}

#last_tp_tracks=np.array([6, 16, 18, 19, 21, 32])
last_tp_tracks=[8, 36, 49, 54, 55, 48, 42, 47, 39, 23, 33, 26, 22]

test=[]
with open(path_files / "Spots_time_ID_id"+str(Cell_ID)+"_toDict.txt" ) as f:
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
                #tp_ID_merge= int(chunks[-1])
                tp_ID_merge= int(chunks[3])
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

data_spots=np.load(path_files / "spots_coords_particle_track_id"+str(Cell_ID)+".npy")
spots_tzyx_nuc_df=pd.read_csv(path_files / "cur_spots_t349_id"+str(Cell_ID)+".csv")
spots_track_coords=pd.DataFrame(data_spots, columns=["ID", "T", "Z", "Y", "X"])
spots_track_coords.ID=spots_track_coords.ID+N_tracks

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
cur_spots.to_csv(path_files / "all_cur_spots_id"+str(Cell_ID)+".csv")

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
cur_spots.to_csv(path_files / "all_cur_spots_id"+str(Cell_ID)+"_w_Merge.csv")


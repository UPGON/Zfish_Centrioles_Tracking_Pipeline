import scipy 
import pandas as pd
import numpy as np


def pairing_points(points1, points2,max_pairing_distance):
    dist_matrix = scipy.spatial.distance_matrix(points1, points2)

    # Optimize only the point that are close enough
    dist_mat_masked = dist_matrix.copy()
    dist_mat_masked[dist_matrix > max_pairing_distance] = 1e10

    opti_row, opti_col = scipy.optimize.linear_sum_assignment(dist_mat_masked)
    distances = dist_matrix[opti_row, opti_col]

    filtered_opti_row = opti_row[distances <= max_pairing_distance]
    filtered_opti_col = opti_col[distances <= max_pairing_distance]
    distances = distances[distances <= max_pairing_distance]

    return filtered_opti_row, filtered_opti_col, distances

    

def pairing_points_df(points1_df, points2_df, max_pairing_distance, columns_name = ["idx1","idx2","dist"]):
    points1 = points1_df[["Zmi", "Ymi", "Xmi"]].values
    points2 = points2_df[["Zmi", "Ymi", "Xmi"]].values

    opti_row, opti_col, distances =  pairing_points(points1, points2, max_pairing_distance)
    
    idx1 = points1_df["index"].iloc[opti_row].values
    idx2 = points2_df["index"].iloc[opti_col].values

    pairing_data = np.stack([idx1,idx2,distances],axis=1)

    return pd.DataFrame(pairing_data, columns=columns_name)


def temporal_pairing_points_df(points1_df, points2_df, max_pairing_distance, columns_name = ["idx1","idx2","dist","T"]):

    results = []

    if(points1_df["T"].max() != points2_df["T"].max()):
         raise ValueError("The points from the dataset should have the same number of frames")

    nbFrames = points1_df["T"].max()

    for frame in range(nbFrames):
        # We should work frame by frame as colocalisation only makes sense for the same temporality
        points1_df_t = points1_df[points1_df["T"] == frame]
        points2_df_t = points2_df[points2_df["T"] == frame]

        opti_row, opti_col, distances = pairing_points(points1_df_t,points2_df_t,max_pairing_distance=max_pairing_distance)

        idx1 = points1_df_t["index"].iloc[opti_row].values
        idx2 = points2_df_t["index"].iloc[opti_col].values

        frame_nb = np.full(len(distances), frame)

        stack_res = np.stack([idx1,idx2,distances,frame_nb],axis=1)
        results.append(stack_res)

    results = np.concatenate(results,axis =0)

    return pd.DataFrame(results,columns=columns_name)

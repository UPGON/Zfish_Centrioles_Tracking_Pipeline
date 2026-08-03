import scipy
import scipy.spatial
import scipy.optimize
import pandas as pd
import numpy as np

def pairing_points(points1, points2, max_pairing_distance, diagonal_pairing=True):
    dist_matrix = scipy.spatial.distance_matrix(points1, points2)

    dist_mat_masked = dist_matrix.copy()
    if not(diagonal_pairing):
        dist_mat_masked[dist_matrix == 0] = 1e10
    dist_mat_masked[dist_matrix > max_pairing_distance] = 1e10

    opti_row, opti_col = scipy.optimize.linear_sum_assignment(dist_mat_masked)
    distances = dist_matrix[opti_row, opti_col]

    valid = distances <= max_pairing_distance
    return opti_row[valid], opti_col[valid], distances[valid]

def pairing_points_df(points1_df, points2_df, max_pairing_distance,
                      columns_name=["idx1", "idx2", "dist"]):

    points1_df = points1_df.reset_index(drop=True)
    points2_df = points2_df.reset_index(drop=True)

    points1 = points1_df[["Zum", "Yum", "Xum"]].values
    points2 = points2_df[["Zum", "Yum", "Xum"]].values

    opti_row, opti_col, distances = pairing_points(points1, points2, max_pairing_distance)

    idx1 = points1_df["index"].iloc[opti_row].values
    idx2 = points2_df["index"].iloc[opti_col].values

    return pd.DataFrame(
        np.stack([idx1, idx2, distances], axis=1),
        columns=columns_name
    )


def temporal_pairing_points_df(points1_df, points2_df, max_pairing_distance,
                                columns_name=["idx1", "idx2", "dist", "T"]):

    frames1 = sorted(points1_df["T"].unique())
    frames2 = sorted(points2_df["T"].unique())

    if frames1 != frames2:
        raise ValueError("The two datasets do not have the same set of frames. "
                         f"Ch1: {frames1}, Ch2: {frames2}")

    results = []

    for ti in frames1:
        points1_df_t = points1_df[points1_df["T"] == ti].reset_index(drop=True)  
        points2_df_t = points2_df[points2_df["T"] == ti].reset_index(drop=True)  

        # Skip frames where one channel has no detections
        if len(points1_df_t) == 0 or len(points2_df_t) == 0:
            print(f"Warning: no spots in frame {ti} for one channel, skipping.")
            continue

        points1 = points1_df_t[["Zum", "Yum", "Xum"]].values
        points2 = points2_df_t[["Zum", "Yum", "Xum"]].values

        opti_row, opti_col, distances = pairing_points(
            points1, points2, max_pairing_distance=max_pairing_distance
        )

        idx1 = points1_df_t["index"].iloc[opti_row].values
        idx2 = points2_df_t["index"].iloc[opti_col].values
        frame_col = np.full(len(distances), ti)

        results.append(np.stack([idx1, idx2, distances, frame_col], axis=1))

    if len(results) == 0:
        return pd.DataFrame(columns=columns_name)

    return pd.DataFrame(np.concatenate(results, axis=0), columns=columns_name)

def pairing_points_anisotropic(points1,points2, max_pairing_distance_xy, max_pairing_distance_z = 2,diagonal_pairing=True):
    dist_matrix = scipy.spatial.distance_matrix(points1, points2)

    dist_mat_masked = dist_matrix.copy()
    if not(diagonal_pairing):
        dist_mat_masked[dist_mat_masked == 0] = 1e10
    max_distances_zyx = np.sqrt(max_pairing_distance_xy**2 + max_pairing_distance_z**2)
    dist_mat_masked[dist_mat_masked > max_distances_zyx] = 1e10

    opti_row, opti_col = scipy.optimize.linear_sum_assignment(dist_mat_masked)

    distances_xy = np.linalg.norm(points1[opti_row,1:3] - points2[opti_col,1:3], axis=1)

    valid_xy = (distances_xy <= max_pairing_distance_xy)

    distances_z = np.abs(points1[opti_row,0] - points2[opti_col,0])
    valid_z = distances_z <= max_pairing_distance_z

    valid = valid_xy & valid_z

    debug = True
    if debug:
        ct_id =273
        cs_id = 122
        print(f"Matching point {opti_row[opti_col == cs_id]}")
        print(f"Distance between the 2 spot in xyz {dist_matrix[ct_id,cs_id]}")
        print(f"Under the xy valid {valid_xy[opti_col == cs_id]}")
        print(f"Under the z valid {valid_z[opti_col == cs_id]}")

        print(f"other candidate distances {dist_matrix[165,cs_id]}")
        print(f"Other candidate match {opti_col[opti_row == 165]}")

    return opti_row[valid], opti_col[valid], distances_xy[valid], distances_z[valid]

def pairing_points_df_anisotropic(points1_df,points2_df, max_pairing_distance_xy, max_pairing_distance_z,
                      columns_name=["idx1", "idx2", "dist_xy","dist_z"]):

    points1_df = points1_df.reset_index(drop=True)
    points2_df = points2_df.reset_index(drop=True)

    points1 = points1_df[["Zum", "Yum", "Xum"]].values
    points2 = points2_df[["Zum", "Yum", "Xum"]].values

    opti_row, opti_col, distances_xy, distances_z = new_pairing(points1, points2, max_pairing_distance_xy, max_pairing_distance_z)


    idx1 = points1_df["index"].iloc[opti_row].values
    idx2 = points2_df["index"].iloc[opti_col].values

    return pd.DataFrame(
        np.stack([idx1, idx2, distances_xy,distances_z], axis=1),
        columns=columns_name
    )


def temporal_pairing_points_df_anisotropic(points1_df,points2_df, max_pairing_distance_xy, max_pairing_distance_z,
                                columns_name=["idx1", "idx2",  "dist_xy","dist_z", "T"]):

    frames1 = sorted(points1_df["T"].unique())
    frames2 = sorted(points2_df["T"].unique())

    if frames1 != frames2:
        raise ValueError("The two datasets do not have the same set of frames. "
                         f"Ch1: {frames1}, Ch2: {frames2}")

    results = []

    for ti in frames1:
        points1_df_t = points1_df[points1_df["T"] == ti].reset_index(drop=True)  
        points2_df_t = points2_df[points2_df["T"] == ti].reset_index(drop=True)  

        # Skip frames where one channel has no detections
        if len(points1_df_t) == 0 or len(points2_df_t) == 0:
            print(f"Warning: no spots in frame {ti} for one channel, skipping.")
            continue

        points1 = points1_df_t[["Zum", "Yum", "Xum"]].values
        points2 = points2_df_t[["Zum", "Yum", "Xum"]].values

        opti_row, opti_col, distances_xy, distances_z =pairing_points_anisotropic(points1, points2, max_pairing_distance_xy, max_pairing_distance_z)


        idx1 = points1_df_t["index"].iloc[opti_row].values
        idx2 = points2_df_t["index"].iloc[opti_col].values
        frame_col = np.full(len(distances_xy), ti)

        results.append(np.stack([idx1, idx2, distances_xy, distances_z, frame_col], axis=1))

    if len(results) == 0:
        return pd.DataFrame(columns=columns_name)

    return pd.DataFrame(np.concatenate(results, axis=0), columns=columns_name)


def new_pairing(points1,points2, max_pairing_distance_xy, max_pairing_distance_z = 2):
    dist_matrix = scipy.spatial.distance_matrix(points1[:,1:3], points2[:,1:3])
    matched1 = []
    matched2 = []
    for i, point1 in enumerate(points1):
        # Get the nearest neighour of my point 1
        z_distance12 = np.abs(point1[0] - points2[:,0])
        candidates2 = np.argwhere(z_distance12 < max_pairing_distance_z)
        closest2 = int(candidates2[dist_matrix[i, candidates2].argmin()][0])

        #Skip this pair if their distance is greater than the threshold
        if dist_matrix[i,closest2] > max_pairing_distance_xy:
            continue

        # Check who is the nearest neighour of my point 2
        z_distance21 = np.abs(points2[closest2,0] - points1[:,0])
        candidates1 = np.argwhere(z_distance21 < max_pairing_distance_z)
        closest1 = candidates1[dist_matrix[candidates1,closest2].argmin()]

        # If both points are the nearest neighour of the other then pair them 
        if i == closest1:
            matched1.append(i)
            matched2.append(closest2)

    distances_xy = np.linalg.norm(points1[matched1,1:3] - points2[matched2,1:3], axis=1)
    distances_z = np.abs(points1[matched1,0] - points2[matched2,0])
    return np.array(matched1), np.array(matched2), distances_xy, distances_z





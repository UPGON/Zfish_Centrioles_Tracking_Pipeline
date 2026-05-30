import scipy
import scipy.spatial
import scipy.optimize
import pandas as pd
import numpy as np


def pairing_points(points1, points2, max_pairing_distance):
    dist_matrix = scipy.spatial.distance_matrix(points1, points2)

    dist_mat_masked = dist_matrix.copy()
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

        idx1      = points1_df_t["index"].iloc[opti_row].values
        idx2      = points2_df_t["index"].iloc[opti_col].values
        frame_col = np.full(len(distances), ti)

        results.append(np.stack([idx1, idx2, distances, frame_col], axis=1))

    if len(results) == 0:
        return pd.DataFrame(columns=columns_name)

    return pd.DataFrame(np.concatenate(results, axis=0), columns=columns_name)
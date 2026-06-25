import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import utils 

# Maximal distance between any label center and the image border in pixel
LABEL_MARGIN = 5

def draw_circles2D(img, centers, radius=[4], color=255, thickness=1):
    """Draw circles on the stack images at the centers coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    if(len(radius) == 1):
        r = np.full(len(centers), radius)
    elif(len(radius) != len(centers)):
        raise ValueError("The number of radius should be either 1 or the same as the number of centers")
    res_img = img.copy()
    for i in range(len(centers)):
        center = centers[i]
        y, x = center.astype(int)
        ri = r[i] if len(radius) > 1 else r[0]
        utils.verif_img_point(img, center)
        cv2.circle(res_img, center=(x, y), radius=ri*2, color=color, thickness=thickness)
    return res_img

def draw_circles(stack, centers, radius=[4], color=255, thickness=1):
    """Draw circles on the stack images at the centers coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    if(len(radius) == 1):
        radius = np.full(len(centers), radius)
    elif(len(radius) != len(centers)):
        raise ValueError("The number of radius should be either 1 or the same as the number of centers")
    res_stack = stack.copy()
    for i in range(len(centers)):
        center = centers[i]
        z, y, x = center.astype(int)
        ri = radius[i].astype(int)
        utils.verif_stack_point(stack, center)
        cv2.circle(res_stack[z], center=(x, y), radius=ri*2, color=color, thickness=thickness)
    return res_stack

def create_circle_mask(centers, stack, radius=[4], color = 200, thickness = 1):
    """Create a mask with circles 

    Args:
        blobs_coords: Array of blob coordinates (z, y, x)
        shape: Shape of the output mask (z, y, x)
        radius: Radius of circles to draw

    Returns:
        3D mask array with circles at blob centers
    """
    mask = np.zeros(stack.shape, dtype=stack.dtype)
    return draw_circles(mask, centers, radius,color, thickness)

def add_text(stack,text, coord, fontScale =0.4,font=cv2.FONT_HERSHEY_COMPLEX_SMALL, text_offset = (-5,2), color = 255):
    """Add text annotation on the stack images at the given coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    res_stack = stack.copy()
    utils.verif_stack_point(stack, coord)
    y_size, x_size = res_stack.shape[-2:]

    text_coord = np.clip(coord[-2:] + text_offset, (LABEL_MARGIN,LABEL_MARGIN), (y_size-LABEL_MARGIN, x_size-LABEL_MARGIN)).astype(int)

    cv2.putText(
        img=res_stack[int(coord[0])],
        text=text,
        org=np.flip(text_coord),
        fontFace=font,
        fontScale=fontScale,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return res_stack

def add_texts(stack,texts,coords,fontScale=0.4, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, text_offset = (-5,2), color = 255):
    """Add text annotation on the stack images at the given coordinates

    Args:
        stack: Input frame of shape [Z,Y,X]
        texts: 
        centers (int,int,int): List of the centers coordinates as [z,y,x]
    """
    res_stack = stack.copy()
    for text, coord in zip(texts, coords):
        res_stack = add_text(res_stack,text, coord, font=font, text_offset=text_offset, color=color,fontScale = fontScale)
    return res_stack

def add_texts_4D(vol, data_df, time_col="T", text_col="idx"):
    t = vol.shape[0]
    tracking_id_mask = np.zeros(vol.shape,dtype = vol.dtype)
    for ti in range(t):  
        tracking_centers = data_df.loc[data_df[time_col]==ti][["Z","Y","X"]].values.astype(int)
        tracking_id = data_df.loc[data_df[time_col]==ti][text_col].values.astype(str)

        tracking_id_mask[ti] = add_texts(tracking_id_mask[ti], tracking_id, tracking_centers, fontScale=0.4)
    return tracking_id_mask

def create_texts_mask(stack,texts,centers, fontScale=0.4,font=cv2.FONT_HERSHEY_COMPLEX_SMALL, text_offset = (-5,2), color = 255):
    mask = np.zeros(stack.shape, dtype=stack.dtype)
    return add_texts(mask, texts, centers,fontScale, font,text_offset, color)

def draw_trajectories(vol, traj_data,track_col="track_id"):
    tracks_ids = traj_data[track_col].unique()

    traj_vol = np.zeros(vol.shape, dtype = vol.dtype)
    for track_i in tracks_ids:
        tracks_objs = traj_data.loc[traj_data[track_col]==track_i][["T","Z","Y","X"]].values.astype(int)

        for i in range(0,len(tracks_objs)):
            if i == 0:
                t,z,y,x = tracks_objs[i]
                traj_vol[t,z] = cv2.circle(traj_vol[t,z],center = (x,y),color = track_i, radius = 2)
            else:
                _,_,prev_y,prev_x = tracks_objs[i-1]
                t,z,y,x = tracks_objs[i]
                
                traj_vol[t,z] = cv2.line(traj_vol[t,z],(x,y),(prev_x,prev_y),color = track_i)
                traj_vol[t,z] = cv2.circle(traj_vol[t,z],center = (x,y),color = track_i, radius = 2)
    return traj_vol
#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  
#  Authors: Red Willow Coleman, willow.coleman@jpl.nasa.gov

import argparse
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio 
from osgeo import gdal
import json
import pandas as pd
import xarray as xr
import cdsapi
from datetime import datetime, timedelta
from time import time
import cv2
import math
from matplotlib import animation
from PIL import Image
from IPython.display import Image, display
from bresenham import bresenham

def main(input_args=None):
    parser = argparse.ArgumentParser(description="Methane plume IME/Q calculation")
    parser.add_argument('--plume_id', type=int, help='Methane plume ID number')
    parser.add_argument('--gif', action='store_true', help="Save plume orthogonal transect gif") 
    parser.add_argument('--plot', action='store_true', help="Save IME/Q vs. plume length scatter plot")
    parser.add_argument('--save_path', type=str, help="Save path for gif/plot outputs", nargs = '?') # Required if gif or plot
    parser.add_argument('--grib_path', type=str, help='Save path for ERA-5 .grib file', default = '/beegfs/scratch/colemanr/emit-ghg/grib', nargs='?')
    args = parser.parse_args(input_args)
    
    if args.save_path is None and args.gif:
        parser.error("--gif and --save_path must be given together")
    if args.save_path is None and args.plot:
        parser.error("--plot and --save_path must be given together")

    ##########################################

    # Path to delineated plume COGs
    plume_path = '/scratch/brodrick/methane/visions_delivery/'

    # Path to plume metadata (.json) 
    metadata_path = '/scratch/brodrick/methane/visions_delivery/combined_plume_metadata.json'

    # Open plume metadata as nested json dict
    f = open(metadata_path)
    metadata = json.load(f)
    plume_df = pd.json_normalize(metadata['features'])

    # Open manual plume annotation for source pixels
    metadata_path_annotation = '/scratch/brodrick/methane/ch4_plumedir/manual_annotation.json'
    f = open(metadata_path_annotation)
    meta = json.load(f)
    meta_df = pd.json_normalize(meta['body']['geojson']['features'])

    # Join meta_df (properties.name) and plume_df (properties.Plume ID) 
    merge_df = pd.merge(plume_df, meta_df, left_on = 'properties.Plume ID', right_on = 'properties.name')

    # Remove duplicate plume metadata "Point" entries 
    poly_df = merge_df[merge_df['geometry.type_x'] == 'Polygon']
    plume_id_list = list(poly_df['properties.Plume ID'])
    scene_fid_list = list(poly_df['properties.Scene FIDs'])
    pseudo_o_list = list(poly_df['properties.Psuedo-Origin'])
    uncertainty_list = list(poly_df['properties.Concentration Uncertainty (ppm m)'])
    
    # Check to see if plume ID exists in json file 
    curr_id = 'CH4_PlumeComplex-' + str(args.plume_id)
    assert curr_id in plume_id_list, f'{"Plume ID "}{str(args.plume_id)}{" does not exist!"}'
    
    i = plume_id_list.index(curr_id)
    json_flag = False
    start = datetime.now()
    plume_id = plume_id_list[i]
    scene_fid = scene_fid_list[i]
    conc_unc = uncertainty_list[i]

    if len(pseudo_o_list[i]) > 0: 
        pseudo_o = json.loads(pseudo_o_list[i])
        json_flag = True

    if len(scene_fid) < 2: 
        for fid in scene_fid:
            curr_folder = os.path.join(fid[4:12], 'l2bch4plm')
            curr_file = fid + '_' + plume_id + '.tif'
            full_path = os.path.join(plume_path, curr_folder, curr_file)

            # Get plume metadata properties
            sub_df = poly_df[poly_df['properties.Plume ID'] == plume_id]
            obs = list(sub_df['properties.UTC Time Observed'])[0]
            day = obs[8:10]
            time = hour_round(obs[11:19])
            year = obs[0:4]
            month = obs[5:7]

            # Load plume as 2D array
            with rasterio.open(full_path, 'r') as ds: 
                plume_arr = ds.read().squeeze()

            if json_flag and len(pseudo_o) > 0: 
                coords = pseudo_o["coordinates"]
                plume_lat = coords[1]
                plume_lon = coords[0]
                x_source, y_source = ~ds.transform * (plume_lon, plume_lat)

            else: 
                print('Using max concentration for plume origin!')
                plume_lat = list(sub_df['properties.Latitude of max concentration'])[0]
                plume_lon = list(sub_df['properties.Longitude of max concentration'])[0]
                x_source, y_source = ~ds.transform * (plume_lon, plume_lat)

    u10_avg, _ = access_era5(time, day, year, month, plume_lat, plume_lon, args.grib_path)
    q_list, ime_list = calc_q(plume_arr, full_path, u10_avg, x_source, y_source, conc_unc, args.plot, args.gif)
    
######## FUNCTIONS ########
    
def access_era5(time, day, year, month, plume_lat, plume_lon, save_path): 
    """
    ## Access ERA5 reanalysis data from CDSAPI (global)
    """
    
    # Plume max concentration plus the nearest pixels on a 0.25 x 0.25 degree scale
    plume_coords = [plume_lat,plume_lon, plume_lat+0.75, plume_lon+0.75, ]
    fname_ext = '.grib'
    fname = year + month + day + '_' + time + fname_ext
    fname = os.path.join(save_path, fname)
    
    # Retrieve ERA5 from Climate Data Store (CDS) API 
    c = cdsapi.Client()
    c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'day': day, 
        'time': time,
        'year': year,
        'month': month,
        'area': plume_coords,
        'variable': [
            '10m_u_component_of_wind',
        ],
    },
    fname)
    
    ds = xr.open_dataset(fname)
    df = ds.to_dataframe()
    u10_avg = np.abs(np.nanmean(df['u10'])) # Positive values = eastward 
    u10_std = np.abs(np.nanstd(df['u10']))
    
    return u10_avg, u10_std

def hour_round(time): 
    """
    ## Round a time string to nearest UTC hour
    """
    
    time_obj = datetime.strptime(time, '%H:%M:%S')
    nearest_hour = (time_obj + timedelta(minutes=30)).replace(minute=0, second=0)
    rounded_time = nearest_hour.strftime('%H:%M:%S')
    return rounded_time[0:-3]

def calc_ime(plume_arr, plume_path, plot = False): 
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    plume_path: filepath to plume enhancement data
    
    ime: excess mass of CH4 in plume [kg]
    p_s: dimensions of EMIT pixel [m^2]
    """

    # Get number of non-NaN pixels in a plume 
    plume_only = plume_arr[plume_arr!=-9999]
    num_pixels = plume_only.size

    #      ppm(m)     L/m^3       mole/L      kg/mole
    k = (1.0/1e6)*(1000.0/1.0)*(1.0/22.4)*(0.01604/1.0) # scaling factor from ppmm to kg CH4

    # IME calculation 
    p_s = pixel_size(plume_path)
    ime = k * np.sum([mf*p_s for mf in plume_only])
    
    if plot: 
        plt.imshow(plume_arr)
        plt.show()
    
    return ime, p_s

def pixel_size(plume_path): 
    """
    plume_path: filepath to plume enhancement data
    
    p_s: dimensions of EMIT pixel [m^2]
    """
    
    proj_ds = gdal.Warp('', plume_path, dstSRS='EPSG:3857', format='VRT')
    transform_3857 = proj_ds.GetGeoTransform()
    xsize_m = transform_3857[1]
    ysize_m = transform_3857[5]
    p_s = np.abs(xsize_m * ysize_m)
    
    return p_s 

def contour_plume(plume_arr, plot = True): 
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    plot: boolean, plot plume and skeleton
    
    right_pt, left_pt: intersect values of skeleton line with x/y axes
    """

    # Scale 0-1
    min_plume = np.min(plume_arr)
    max_plume = np.max(plume_arr)
    scale_plume = (plume_arr - min_plume)/(max_plume - min_plume)
    scale_plume = (scale_plume * 255).astype(np.uint8)

    # Create binary threshold of plume/background
    img = 255 - scale_plume
    a = img.max()  
    _, thresh = cv2.threshold(img, a/2+60, a,cv2.THRESH_BINARY_INV)

    # Find the contour of the plume 
    contours, hierarchy = cv2.findContours(
                                       image = thresh, 
                                       mode = cv2.RETR_TREE, 
                                       method = cv2.CHAIN_APPROX_NONE)

    # Sort the contours 
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    c = contours[0]

    # Calculate line through plume
    img_copy = np.stack((img,)*3, axis = -1) # create a 3-channel image for displaying contour
    rows,cols = img_copy.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
    left_pt = int((-x*vy/vx) + y)
    right_pt = int(((cols-x)*vy/vx)+y)
    curr_line = cv2.line(img_copy,(cols-1,right_pt),(0,left_pt),(0,0,255),1)
    line_length = math.dist([cols-1,right_pt],[0,left_pt])

    # Plotting
    if plot: 
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        cv2.circle(img_copy, (cX, cY), 1, (255, 0, 0), -1)
        cv2.drawContours(image=img_copy, contours=contours, 
                         contourIdx=-1, color=(0, 255, 0), thickness=1)
        plt.imshow(img_copy, cmap = 'gray')
        plt.show()
        
    right_pt = [cols-1,right_pt]
    left_pt = [0,left_pt]

    return line_length, right_pt, left_pt
    
def find_intersection(plume_arr, row_index, col_index):
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    row_index: x coordinate of point of interest
    col_index: y coordinate of point of interest
    
    intersection_points: list of points along line from row_index, col_index to edges of plume_arr
    """
    
    # Find plume contour line and slope 
    line_length, right_pt, left_pt = contour_plume(plume_arr, plot = False)
    slope = (right_pt[1] - left_pt[1])/(right_pt[0] - left_pt[0])
    slope = -1/slope
    
    rows, cols = plume_arr.shape
    b = row_index - slope * col_index
    intersection_points = []

    # Calculate the possible intersection points with the top and bottom edges
    x_top = (0 - b) / slope
    x_bottom = (rows - 1 - b) / slope

    # Check if the intersection points are within the array bounds
    if 0 <= x_top < cols:
        intersection_points.append((int(x_top), 0))
    if 0 <= x_bottom < cols:
        intersection_points.append((int(x_bottom), rows - 1))

    # Calculate the possible intersection points with the left and right edges
    y_left = slope * 0 + b
    y_right = slope * (cols - 1) + b

    # Check if the intersection points are within the array bounds
    if 0 <= y_left < rows:
        intersection_points.append((0, int(y_left)))
    if 0 <= y_right < rows:
        intersection_points.append((cols - 1, int(y_right)))

    return intersection_points

def calc_q(plume_arr, plume_path, u10, x_source, y_source, conc_unc, plot, gif, gif_path = 'ime_q.gif'): 
    """
    plume_arr [float arr]: 2-D array of plume enhancement data [ppmm]
    plume_path [str]: filepath to plume enhancement data
    u10 [float]: 10 m near surface windspeed from re-analysis data 
    x_source: x coordinate of plume source pixel
    y_source: y coordinate of plume source pixel
    conc_unc: scene-wise match-filter concentration uncertainty [ppmm] 
    plot: boolean, scatterplot of IME and Q values
    gif: boolean, gif of orthogonally transected plume 
    
    q [float]: hourly plume emissions [kgCH4/hr]
    ime [float]: excess mass of CH4 in plume [kg]
    """
    
    # Get locations of plume skeleton line 
    line_length, right_pt, left_pt = contour_plume(plume_arr, plot = False)

    # Rasterize line using Bresenham Line Algorithm
    indices = list(bresenham(right_pt[0], right_pt[1], left_pt[0], left_pt[1]))
    col_inds = [item[0] for item in indices]
    row_inds = [item[1] for item in indices]

    # Remove out-of-bounds skeleton indices
    del_x_vals = [index for index, item in enumerate(row_inds) if item < 0 or item > plume_arr.shape[0]]
    for index in sorted(del_x_vals, reverse=True):
        del row_inds[index]
        del col_inds[index]
    del_y_vals = [index for index, item in enumerate(col_inds) if item < 0 or item > plume_arr.shape[1]]
    for index in sorted(del_y_vals, reverse=True):
        del row_inds[index]
        del col_inds[index]
        
    # For each (row,ind) pair in the skeleton, orthogonally bisect the widest part 
    q_list = []
    rad_list = []
    ime_list = []
    q_unc_list = []
    ime_unc_list = []
    frames = []
    
    if gif: 
        fig, ax = plt.subplots()

    # Determine whether source pixel is closer to left/right axis
    num_rows, num_cols = plume_arr.shape
    left_distance_col = x_source
    right_distance_col = num_cols - x_source - 1
        
    # Iterate through row/col inds 
    for i in range(0,len(row_inds), 5):
        row = row_inds[i]
        col = col_inds[i]
        
        # Find orthogonal line 
        intersection_points = find_intersection(plume_arr, row, col)
        x1 = intersection_points[0][1]
        y1 = intersection_points[0][0]
        x2 = intersection_points[1][1]
        y2 = intersection_points[1][0]
        indices = list(bresenham(x1, y1, x2, y2))
        
        # Calculate the slope and y-intercept of the line
        m = (y2 - y1) / ((x2 - x1) + 10e-6)
        b = y1 - m * x1

        # Mask out based on orthogonal transect line 
        curr_plume_arr = plume_arr.copy()        
        x_indices, y_indices = np.indices(plume_arr.shape)
        if left_distance_col <= right_distance_col: # Source closer to left, mask to the right 
            mask = y_indices >= (m * x_indices + b)
        else: # Source closer to right, mask to the left
            mask = y_indices <= (m * x_indices + b)
        curr_plume_arr[mask] = -9999 
        
        # Mask out all negative MF values 
        curr_plume_arr = np.where(curr_plume_arr < 0, -9999, curr_plume_arr)                        
        
        # Only include subplumes where the source pixel is in the plume
        if curr_plume_arr[int(y_source), int(x_source)] != -9999:
            if gif:
                # Save for gifs 
                curr_fig = ax.imshow(curr_plume_arr, cmap = 'viridis')
                point = ax.scatter(x_source, y_source, marker = 'x', color = 'red')
                frames.append([point, curr_fig])

            # IME/Q calculations
            ime, p_s = calc_ime(curr_plume_arr, plume_path)
            plume_length = np.abs(math.dist([y_source, x_source],[row,col])) * np.sqrt(p_s)
            q = (ime/plume_length) * u10 * 3600 
            
            # IME/Q uncertainty  
            unc_mask = curr_plume_arr != -9999
            unc_plume_arr = curr_plume_arr.copy()
            unc_plume_arr[unc_mask] = conc_unc         
            ime_unc, p_s = calc_ime(unc_plume_arr, plume_path)
            q_unc = (ime_unc/plume_length) * u10 * 3600 

            q_list.append(q)
            rad_list.append(plume_length)
            ime_list.append(ime)
            ime_unc_list.append(ime_unc)
            q_unc_list.append(q_unc)
     
    if gif: 
        # Display animation of concentric circles 
        ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=1000)
        ani.save(gif_path)
        display(Image(gif_path))
        
    if plot: 
        # Plot IME/Q data
        fig1, ax1 = plt.subplots()
        ax1.errorbar(rad_list, q_list, yerr = q_unc_list, color = 'blue', fmt = 'o', capsize = 2)
        ax1.set_xlabel('Plume length [m]')
        ax1.set_ylabel('Q [kgCH4/hr]', color = 'blue')
        ax1.tick_params(axis = 'y', labelcolor = 'blue')
        ax2 = ax1.twinx()
        ax2.errorbar(rad_list, ime_list, yerr = ime_unc_list, color = 'r', fmt = 'o', capsize = 2)
        ax2.set_ylabel('IME [kg]', color = 'r')
        ax2.tick_params(axis = 'y', labelcolor = 'r')
        plt.show()
        
        # Plot IME/Q uncertainty data
        fig1, ax1 = plt.subplots()
        ax1.scatter(rad_list, [a / b for a, b in zip(ime_list, ime_unc_list)], color = 'green')
        ax1.set_xlabel('Plume length [m]')
        ax1.set_ylabel('IME/IME Uncertainty')
        ax1.axhline(y=1, color = 'green', linestyle = 'dashed', linewidth = 2)

        plt.show()

    return q_list, ime_list

    
if __name__ == '__main__':
    main()

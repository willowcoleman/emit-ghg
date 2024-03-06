#! /usr/bin/env python
#
#  Copyright 2024 California Institute of Technology
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
from datetime import datetime, timedelta
from time import time
import cv2
import math
from bresenham import bresenham
import openmeteo_requests
import requests_cache
from retry_requests import retry
from openmeteo_sdk.Variable import Variable
from LatLongUTMconversion import LLtoUTM

def main(input_args=None):
    parser = argparse.ArgumentParser(description="Methane plume IME/Q calculation")
    parser.add_argument('--plume_id', type=int, help='Methane plume ID number')
    parser.add_argument('--method', type=str, default='transect', help="??")
    parser.add_argument('--plume_path', type=str, default='/scratch/brodrick/methane/visions_delivery/', help="Path to delineated plume COGs")
    parser.add_argument('--meta_path', type=str, default='/scratch/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json', help="Path to metadata json")
    parser.add_argument('--mask_neg', action='store_true')
    parser.add_argument('--plot', action='store_true', help="Save IME/Q vs. plume length scatter plot figure as pdf")
    parser.add_argument('--plot_path', type=str, default='/scratch/colemanr/emit-ghg/ime_2024_manuscript/transect_figs/', help="Save path for pdf plot outputs") # Required if gif or plot

    parser.add_argument('--plume_lat', type=float)
    parser.add_argument('--plume_lon', type=float)

    args = parser.parse_args(input_args)

    # Flags
    if args.mask_neg: 
        print('Masking out negative MF values')
    
    # Check for co-required arguments
    if args.plot_path is None and args.plot:
        parser.error("--plot and --plot_path must be given together")
        
    # Check that provided paths exist and create if they don't 
    if args.plot_path is not None and not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
        print("Directory '% s' created" % args.plot_path) 

    ##########################################

    # Extract plume metadata 
    full_plume_id = 'CH4_PlumeComplex-' + str(args.plume_id)
    date, hour_rounded, plume_lat, plume_lon, emit_fid, full_path = get_plume_metadata_MMGIS(full_plume_id, args.meta_path, args.plume_path)
    u10_avg, u10_std = open_meteo_era5(plume_lat, plume_lon, date, hour_rounded)

    # Read plume array from .tif 
    plume_arr, x_source, y_source = read_plume_arr(full_path, plume_lat, plume_lon)

    # IME/Q Calculations 
    if args.method == 'transect':
        p_s = pixel_size(full_path, plume_lat, plume_lon)
        q_list, ime_list, rad_list = calc_q_transect(plume_arr, p_s, u10_avg, u10_std, x_source, y_source, args.mask_neg)

        if args.plot_path: 
            create_plume_pdf(plume_arr, x_source, y_source, rad_list, ime_list, q_list, plume_lat, plume_lon, full_plume_id, emit_fid, args.plot_path)
    else: 
        print("This method is not implemented!")
        return

######## HELPER FUNCTIONS ########

def read_plume_arr(full_path, plume_lat, plume_lon):
    """
    full_path: complete path to individual delineated plume .tif file
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]

    plume_arr: 2-D array of plume enhancement data [ppmm]
    x_source: x coordinate of plume source pixel 
    y_source: y coordinate of plume source pixel
    """
    with rasterio.open(full_path, 'r') as ds: 
        plume_arr = ds.read().squeeze()

    # Convert source to pixel coordinates
    x_source, y_source = ~ds.transform * (plume_lon, plume_lat)

    return plume_arr, x_source, y_source

def get_plume_metadata_MMGIS(curr_id, meta_path, plume_path):
    """
    curr_id: plume ID string (e.g., CH4_PlumeComplex-1800)
    meta_path: path to metadata json file scraped from MMGIS
    plume_path: path to folder of delineated plume .tif files 

    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    date: str year-month-day e.g., "2024-03-05"
    hour_rounded: e.g., 19:00:00
    emit_fid: EMIT acquisition FID (e.g., emit20230219t094130)
    full_path: complete path to individual delineated plume .tif file
    """

    # MMGIS metadata 
    with open(meta_path, 'r') as f: 
        meta = json.loads(f.read())
    meta_df = pd.json_normalize(meta['features'])
    meta_df = meta_df[['properties.Plume ID', 'properties.fids', 'properties.Psuedo-Origin']]

    # Check to see if plume ID exists in json file 
    assert curr_id in list(meta_df['properties.Plume ID']), f'{str(curr_id)}{" does not exist!"}'

    # Extract metadata for current complex_id
    metadata = meta_df.iloc[meta_df[meta_df['properties.Plume ID'].str.contains(curr_id.strip())].index[0]]

    # Check for missing pseudo-origin data 
    if not (metadata['properties.Psuedo-Origin']):
        print(curr_id + ' missing pseudo-origin data!\n')
        return 

    else: 
        # Read out metadata 
        pseudo_o  = json.loads(metadata['properties.Psuedo-Origin'])["coordinates"]
        plume_lat = pseudo_o[1]
        plume_lon = pseudo_o[0]
        scene_fid = metadata['properties.fids']
        date = scene_fid[0][4:8] + "-" + scene_fid[0][8:10] + "-" + scene_fid[0][10:12]
        time_str = scene_fid[0][13:15] + ":" + scene_fid[0][15:17] + ":" + scene_fid[0][17:19]
        hour_rounded = hour_round(time_str)

        # Open plume tifs, checking for plumes that may span multiple FIDs but are contained in a single plume .tif
        for fid in scene_fid:
            curr_folder = os.path.join(fid[4:12], 'l2bch4plm')
            curr_file = fid + '_' + curr_id.strip() + '.tif'
            full_path = os.path.join(plume_path, curr_folder, curr_file)

            if os.path.isfile(full_path): 
                return date, hour_rounded, plume_lat, plume_lon, fid, full_path

def open_meteo_era5(plume_lat, plume_lon, date, hour_rounded): 
    """
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    date: str year-month-day e.g., "2024-03-05"
    hour_rounded: e.g., 19:00:00

    u10_avg: average 10 m windspeed over 0.75 x 0.75 deg square [m/s]
    u10_std: standard deviation 10 m windspeed over 0.75 x 0.75 deg square [m/s]
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Call Open-Meteo API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": [plume_lat-0.25, plume_lat, plume_lat+0.25, 
                plume_lat-0.25, plume_lat, plume_lat+0.25, 
                plume_lat-0.25, plume_lat, plume_lat+0.25],
        "longitude": [plume_lon+0.25, plume_lon+0.25, plume_lon+0.25,
                  plume_lon, plume_lon, plume_lon, 
                  plume_lon-0.25, plume_lon-0.25, plume_lon-0.25],
        "start_date": date,
        "end_date": date,
        "hourly": "wind_speed_10m",
        "models": "best_match", 
        "wind_speed_unit": "ms"
    }
    responses = openmeteo.weather_api(url, params=params)
    
    wind_list = []
    for response in responses: 
        hourly = response.Hourly()
        hourly_time = range(hourly.Time(), hourly.TimeEnd(), hourly.Interval())
        hourly_variables = list(map(lambda i: hourly.Variables(i), range(0, hourly.VariablesLength())))

        # Extract hourly wind speed for specified hour 
        hourly_wind_speed_10m = next(filter(lambda x: x.Variable() == Variable.wind_speed and x.Altitude() == 10, hourly_variables)).ValuesAsNumpy()
        hour_ind = int(hour_rounded.split(":")[0])
        wind_list.append(hourly_wind_speed_10m[hour_ind])
    
    u10_avg = np.abs(np.nanmean(wind_list))
    u10_std = np.abs(np.nanstd(wind_list))
    
    return u10_avg, u10_std

def create_plume_pdf(plume_arr, x_source, y_source, rad_list, ime_list, q_list, plume_lat, plume_lon, full_plume_id, emit_fid, plot_path):
    """
    Write pdf that displays plume delineation, IME/Q scatterplots 

    plume_arr: 2-D array of plume enhancement data [ppmm]
    x_source: x coordinate of plume source pixel 
    y_source: y coordinate of plume source pixel
    q_list: list of hourly plume emissions per transect [kgCH4/hr]
    ime_list: list of excess mass of CH4 in plume per transect[kg]
    rad_list: list of sub-transect fetch lengths from plume pseudo-origin [m]
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    emit_fid: EMIT acquisition FID (e.g., emit20230219t094130)
    plot_path: save path for pdf plot outputs
    """

    fig, ax = plt.subplots(1,2,figsize = (10,4))
    
    plume_img = np.where(plume_arr == -9999 , np.nan, plume_arr)
    ax[0].imshow(plume_img, vmin = 0, vmax = 1500, cmap = 'plasma')
    ax[0].plot(x_source, y_source, marker='o', mfc='white', color = 'black', linestyle='', ms = 7)
    ax[1].scatter(rad_list, ime_list, color = 'red')
    ax[1].set(xlabel = 'Plume length [m]', ylabel = 'IME [kg]')
    ax[1].tick_params(axis = 'y', color = 'red', labelcolor = 'red')
    ax[1].set_ylabel('IME [kg]', color = 'red')
    ax2 = ax[1].twinx()
    ax2.scatter(rad_list, q_list, color = 'blue')
    ax2.tick_params(axis = 'y', color = 'blue', labelcolor = 'blue')
    ax2.set_ylabel('Q [kgCH4/hr]', color = 'blue')

    fig.tight_layout(pad=2.0)
    fig.suptitle(full_plume_id + '\n\n' + emit_fid + '\n Pseudo-origin: (' + str(round(plume_lat,3)) + ', ' + str(round(plume_lon,3)) + ')', y=1.2, x = 0.5, ha = 'center')                    
    fig_path = os.path.join(plot_path, full_plume_id.strip() + '_transect.pdf')
    fig.savefig(fig_path, bbox_inches='tight')

    plt.close()

def hour_round(time): 
    """
    Round a time string to nearest UTC hour
    """
    
    time_obj = datetime.strptime(time, '%H:%M:%S')
    nearest_hour = (time_obj + timedelta(minutes=30)).replace(minute=0, second=0)
    rounded_time = nearest_hour.strftime('%H:%M:%S')
    
    return rounded_time[0:-3]

def calc_ime_transect(plume_arr, p_s): 
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    p_s: dimensions of EMIT pixel [m^2]    

    ime: excess mass of CH4 in plume [kg]
    """

    # Get non-NaN pixels in a plume 
    plume_only = plume_arr[plume_arr!=-9999]
    
    # IME calculation     
    #     ppm(m)       m^2       L/m^3        mol/L      kg/mol
    k = (1.0/1e6)*((p_s)/1.0)*(1000.0/1.0)*(1.0/22.4)*(0.01604/1.0) # scaling factor from ppmm to kg CH4
    ime = plume_only.sum() * k 
    
    return ime

def pixel_size(full_path, plume_lat, plume_lon):
    """
    full_path: complete path to individual delineated plume .tif file
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]

    p_s: dimensions of EMIT pixel [m^2]    
    """

    # Extract EPSG code at pseudo-origin
    utm_zone,_,_ = LLtoUTM(23, plume_lat, plume_lon)
    epsg_code = 32600
    epsg_code += int(utm_zone[:-1])
    if plume_lat < 0:
        epsg_code += 100
    dstSRS = 'EPSG:' + str(epsg_code)

    # Convert from deg to m 
    proj_ds = gdal.Warp('', full_path, dstSRS=dstSRS, format='VRT')
    transform_ds = proj_ds.GetGeoTransform()
    xsize_m = transform_ds[1]
    ysize_m = transform_ds[5]
    p_s = np.abs(xsize_m * ysize_m)

    return p_s

def contour_plume(plume_arr, plot = False): 
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
    line = ((0, left_pt), (img_copy.shape[1] - 1, right_pt))
    
    # Clip line to contour 
    rect = cv2.boundingRect(c)
    x, y, w, h = rect
    clipped_line = cv2.clipLine(rect, line[0], line[1])
    cv2.line(img_copy, clipped_line[1], clipped_line[2], (0, 255, 0), 1)
    
    # Plotting
    if plot: 
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        cv2.drawContours(image=img_copy, contours=contours, 
                         contourIdx=-1, color=(0, 255, 0), thickness=1)
        cv2.line(img_copy, clipped_line[1], clipped_line[2], (0, 0, 255), 1)
        plt.imshow(img_copy, cmap = 'gray')
        plt.show()
        
    # Return calculations     
    right_pt = clipped_line[2]
    left_pt = clipped_line[1]
    line_length = math.dist([clipped_line[2][0], clipped_line[2][1]], [clipped_line[1][0], clipped_line[1][1]])
    
    return line_length, right_pt, left_pt
    
def find_intersection(plume_arr, row_index, col_index):
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    row_index: x coordinate of point of interest
    col_index: y coordinate of point of interest
    
    intersection_points: list of points along line from row_index, col_index to edges of plume_arr
    """
    
    # Find plume contour line and slope 
    line_length, right_pt, left_pt = contour_plume(plume_arr)
        
    slope = (right_pt[1] - left_pt[1])/(right_pt[0] - left_pt[0])
    slope = -1/(slope + 10e-6)
    
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

def calc_q_transect(plume_arr, p_s, u10, u10_std, x_source, y_source, mask_neg, gif = True): 
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    p_s: dimensions of EMIT pixel [m^2]    
    u10_avg: average 10 m near surface windspeed [m/s]
    u10_avg: standard deviation 10 m near surface windspeed [m/s]
    x_source: x coordinate of plume source pixel 
    y_source: y coordinate of plume source pixel
    gif [boolean]: gif of orthogonally transected plume 
    
    q_list: list of hourly plume emissions per transect [kgCH4/hr]
    ime_list: list of excess mass of CH4 in plume per transect[kg]
    rad_list: list of sub-transect fetch lengths from plume pseudo-origin [m]
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
    q_list, rad_list, ime_list, frames = [], [], [], []
        
    if gif: 
        fig, ax = plt.subplots()

    # Determine whether source pixel is closer to left/right axis
    num_rows, num_cols = plume_arr.shape
    left_distance_col = x_source
    right_distance_col = num_cols - x_source - 1
        
    # Iterate through row/col inds 
    for i in range(0,len(row_inds), 1):
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
        if mask_neg: 
            curr_plume_arr = np.where(curr_plume_arr < 0, -9999, curr_plume_arr)
        
        # Only include subplumes where the source pixel is in the plume
        if curr_plume_arr[int(y_source), int(x_source)] != -9999:
            if gif:
                # Save for gifs 
                curr_fig = ax.imshow(curr_plume_arr, cmap = 'viridis')
                point = ax.scatter(x_source, y_source, marker = 'x', color = 'red')
                frames.append([point, curr_fig])

            # IME/Q calculations
            ime = calc_ime_transect(curr_plume_arr, p_s)
            plume_length = np.abs(math.dist([y_source, x_source],[row,col])) * np.sqrt(p_s)
            q = (ime/plume_length) * u10 * 3600          

            q_list.append(q)
            rad_list.append(plume_length)
            ime_list.append(ime)            

    return q_list, ime_list, rad_list

if __name__ == '__main__':
    main()

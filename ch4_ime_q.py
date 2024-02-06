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
import re

def main(input_args=None):
    parser = argparse.ArgumentParser(description="Methane plume IME/Q calculation")
    parser.add_argument('--plume_id', type=int, help='Methane plume ID number')
    parser.add_argument('--out_path', type=str, default = '/beegfs/scratch/colemanr/emit-ghg/out/', nargs='?', help='Save path for output log file')
    parser.add_argument('--out_name', type=str, default='save_data.txt', help='Name for output log file (.txt)')
    parser.add_argument('--gif', action='store_true', help="Save plume orthogonal transect gif") 
    parser.add_argument('--plot', action='store_true', help="Save IME/Q vs. plume length scatter plot")
    parser.add_argument('--plot_path', type=str, help="Save path for gif/plot outputs") # Required if gif or plot
    parser.add_argument('--grib_path', type=str, help='Save path for ERA-5 .grib file', default = '/beegfs/scratch/colemanr/emit-ghg/grib', nargs='?')
    args = parser.parse_args(input_args)
    
    # Check for co-required arguments
    if args.plot_path is None and args.gif:
        parser.error("--gif and --plot_path must be given together")
    if args.plot_path is None and args.plot:
        parser.error("--plot and --plot_path must be given together")
        
    # Check that provided paths exist and create if they don't 
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print("Directory '% s' created" % args.out_path) 
    if args.plot_path is not None and not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
        print("Directory '% s' created" % args.plot_path) 
    if not os.path.exists(args.grib_path):
        os.makedirs(args.grib_path)
        print("Directory '% s' created" % args.grib_path) 
    
    # Check that args.out_name is a .txt file and if it doesn't exist yet, create it
    if os.path.splitext(args.out_name)[1] != '.txt':
        parser.error("--out_name must be a .txt file")
    if not os.path.isfile(os.path.join(args.out_path, args.out_name)):
        with open(os.path.join(args.out_path, args.out_name), 'w') as f:
            print("File '% s' created" % os.path.join(args.out_path, args.out_name))
            f.write('')  
    else: 
        print("Appending data to existing file!") 

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

    # IME/Q Calculations 
    u10_avg, u10_std = access_era5(time, day, year, month, plume_lat, plume_lon, args.grib_path)
    q_list, ime_list, min_radius = calc_q(plume_arr, full_path, u10_avg, u10_std, args.plume_id, x_source, y_source, conc_unc, args.plot, args.gif, args.plot_path)
    
    print('Threshold:', str(min_radius))
    
    plume_area = np.sum(plume_arr>0)
    
    # Write plume info to file
    with open(os.path.join(args.out_path, args.out_name), 'a+', encoding='utf-8') as my_file:
        my_file.write('Plume ID: ' + str(plume_id) + '\n')
        my_file.write('Threshold: ' + str(min_radius) + '\n')
        my_file.write('Plume Area: ' + str(plume_area) + '\n')
        my_file.write('Max IME: ' + str(np.max(ime_list)) + '\n')
        my_file.write('Max Q: ' + str(np.max(q_list)) + '\n')
        my_file.write('Avg Windspeed: ' + str(u10_avg) + '\n')
        my_file.write('Windspeed Unc: ' + str(u10_std) + '\n')
        

    
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

    if not os.path.exists(fname): 
        print('Downloading new .grib file!')
    
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
    else:
        print('Using pre-existing .grib file!')
    
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

def calc_ime(plume_arr, plume_path, dst_srs, plot = False): 
    """
    plume_arr: 2-D array of plume enhancement data [ppmm]
    plume_path: filepath to plume enhancement data
    
    ime: excess mass of CH4 in plume [kg]
    p_s: dimensions of EMIT pixel [m^2]
    """

    # Get non-NaN pixels in a plume 
    plume_only = plume_arr[plume_arr!=-9999]
    
    # IME calculation 
    p_s = pixel_size(plume_path, dst_srs)
    
    #     ppm(m)       m^2       L/m^3        mol/L      kg/mol
    k = (1.0/1e6)*((p_s)/1.0)*(1000.0/1.0)*(1.0/22.4)*(0.01604/1.0) # scaling factor from ppmm to kg CH4
    ime = plume_only.sum() * k 
    
    if plot: 
        plt.imshow(plume_arr)
        plt.show()
    
    return ime, p_s

def pixel_size(plume_path, dst_srs): 
    """
    plume_path: filepath to plume enhancement data
    
    p_s: area dimensions of EMIT pixel [m^2]
    """
    
    proj_ds = gdal.Warp('', plume_path, dstSRS=dst_srs, format='VRT')
    transform_ds = proj_ds.GetGeoTransform()
    xsize_m = transform_ds[1]
    ysize_m = transform_ds[5]
    p_s = np.abs(xsize_m * ysize_m)
    
    return p_s 

def contour_plume_UPDATE(plume_arr, plot = True): 
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
    line_length, right_pt, left_pt = contour_plume_UPDATE(plume_arr, plot = False)
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

def calc_q(plume_arr, plume_path, u10, u10_std, complex_id, x_source, y_source, conc_unc, plot, gif, plot_path, dst_srs = 'EPSG:3857'): 
    """
    plume_arr [float arr]: 2-D array of plume enhancement data [ppmm]
    plume_path [str]: filepath to plume enhancement data
    u10 [float]: 10 m near surface windspeed from re-analysis data 
    x_source: x coordinate of plume source pixel
    y_source: y coordinate of plume source pixel
    conc_unc: scene-wise match-filter concentration uncertainty [ppmm] 
    plot [boolean]: scatterplot of IME and Q values
    gif [boolean]: gif of orthogonally transected plume 
    
    q [float]: hourly plume emissions [kgCH4/hr]
    ime [float]: excess mass of CH4 in plume [kg]
    """
    
    # Get locations of plume skeleton line 
    line_length, right_pt, left_pt = contour_plume_UPDATE(plume_arr, plot = False)

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
        # curr_plume_arr = np.where(curr_plume_arr < 0, -9999, curr_plume_arr)
        
        # Only include subplumes where the source pixel is in the plume
        if curr_plume_arr[int(y_source), int(x_source)] != -9999:
            if gif:
                # Save for gifs 
                curr_fig = ax.imshow(curr_plume_arr, cmap = 'viridis')
                point = ax.scatter(x_source, y_source, marker = 'x', color = 'red')
                frames.append([point, curr_fig])

            # IME/Q calculations
            ime, p_s = calc_ime(curr_plume_arr, plume_path, dst_srs)
            plume_length = np.abs(math.dist([y_source, x_source],[row,col])) * np.sqrt(p_s)
            q = (ime/plume_length) * u10 * 3600 
            
            # IME/Q uncertainty  
            unc_mask = curr_plume_arr != -9999
            unc_plume_arr = curr_plume_arr.copy()
            unc_plume_arr[unc_mask] = conc_unc         
            ime_unc, p_s = calc_ime(unc_plume_arr, plume_path, dst_srs)
            q_unc = (ime_unc/plume_length) * u10 * 3600 

            q_list.append(q)
            rad_list.append(plume_length)
            ime_list.append(ime)
            ime_unc_list.append(ime_unc)
            q_unc_list.append(q_unc)
     
    # When is scenewide uncertainty >> than measured MF in ppmm 
    plume_threshold_list = [a / b for a, b in zip(ime_list, ime_unc_list)]
    
    # Find highest value in rad_list where IME/IME uncertainty > 1 
    min_radius = 0
    for i in range(len(plume_threshold_list)): 
        radius = rad_list[i]
        thresh = plume_threshold_list[i]
        if radius > min_radius and thresh > 1: 
            min_radius = radius
    
    # # Get plume complex ID 
    # complex_id = re.search(r'-(.*?)\.tif', os.path.split(plume_path)[-1]).group(1)

    if gif: 
        # Display animation of concentric circles 
        ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=1000)
        ani.save(os.path.join(plot_path,  str(complex_id) + '_plume_ani.gif'))
        # display(Image(plot_path))
        
    if plot: 
        
        plt.rcParams['font.family'] = "serif"
        
        # fig1, (ax1, ax3) = plt.subplots(1,2, figsize = (10, 4))
        fig1, ax1 = plt.subplots(1,1, figsize = (5,4))
        
        fig1.tight_layout(pad=7)
        
        # Plot IME/Q data
        ax1.scatter(rad_list, q_list, color = 'blue')
        ax1.set_xlabel('Plume length [m]')
        ax1.set_ylabel('Q [kgCH4/hr]', color = 'blue')
        ax1.tick_params(axis = 'y', labelcolor = 'blue')
        ax2 = ax1.twinx()
        ax2.scatter(rad_list, ime_list, color = 'r')
        ax2.set_ylabel('IME [kg]', color = 'r')
        ax2.tick_params(axis = 'y', labelcolor = 'r')
        plt.savefig(os.path.join(plot_path, str(complex_id) + '_ime_q_scatter.png'), dpi=1200)
        
        ## Read in CNN bypass data 
        cnn_df = pd.read_csv('/scratch/colemanr/emit-ghg/EMIT_plume_ime_CNN.csv')
        
        ime_200 = np.mean(cnn_df[cnn_df[' Candidate ID'] == ' CH4_PlumeComplex-' + str(complex_id)][' IME200 (kg)'])
        # ax2.plot([200], [ime_200], marker='*', ls='none', color = 'r', ms=10)
        
        avg_ime_200 = np.mean(cnn_df[cnn_df[' Candidate ID'] == ' CH4_PlumeComplex-' + str(complex_id)][' AvgIMEdivFetch200 (kg/m)'])
        stdev = np.mean(cnn_df[cnn_df[' Candidate ID'] == ' CH4_PlumeComplex-' + str(complex_id)][' StdIMEdivFetch200 (kg/m)'])
        
        ime_200 = np.mean(cnn_df[cnn_df[' Candidate ID'] == ' CH4_PlumeComplex-' + str(complex_id)][' IME200 (kg)'])
        fetch_200 = np.mean(cnn_df[cnn_df[' Candidate ID'] == ' CH4_PlumeComplex-' + str(complex_id)][' Fetch200 (m)'])
        
        # Science Advances 2023
#         ax1.axhline([(ime_200/fetch_200) * u10 * 3600], ls='solid', color = 'blue')
#         ax1.axhline([(ime_200/fetch_200) * u10 * 3600 + (ime_200/fetch_200) * u10_std * 3600], ls='dashed', color = 'blue')
#         ax1.axhline([(ime_200/fetch_200) * u10 * 3600 - (ime_200/fetch_200) * u10_std * 3600], ls='dashed', color = 'blue')
        
        # Duren Concentric Circles 2019
        # ax1.axhline([avg_ime_200 * u10 * 3600], ls='solid', color = 'blue')
        # ax1.axhline([avg_ime_200 * u10 * 3600 + stdev * 3600 * u10], ls='dashed', color = 'blue')
        # ax1.axhline([avg_ime_200 * u10 * 3600 - stdev * 3600 * u10], ls='dashed', color = 'blue')
        
        # # Plot IME/Q uncertainty data
        # # fig1, ax1 = plt.subplots()
        # ax3.scatter(rad_list, plume_threshold_list, color = 'green')
        # ax3.set_xlabel('Plume length [m]')
        # ax3.set_ylabel('IME/IME Uncertainty')
        # ax3.axhline(y=1, color = 'green', linestyle = 'dashed', linewidth = 2)
        # # plt.savefig(os.path.join(plot_path, str(complex_id) + '_ime_q_scatter_unc.png'))
        
        plt.show()


    return q_list, ime_list, rad_list

    
if __name__ == '__main__':
    main()

import argparse
from herbie import Herbie
import xarray as xr
import numpy as np
import pandas as pd 
import numpy as np
import glob 
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
from openmeteo_sdk.Variable import Variable
from emit_utils.file_checks import envi_header
from spectral.io import envi


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Retrieve HRRR, ERA5, ECMWF U10 reanalysis data")
    parser.add_argument('--plume_lat', type=float,  help='Latitude of plume pseudo-origin or desired windspeed location')
    parser.add_argument('--plume_lon', type=float,  help='Longitude of plume pseudo-origin or desired windspeed location')
    parser.add_argument('--fid', type=str, help='')

    args = parser.parse_args(input_args)
    
    u10_hrrr, u10_hrrr_stddev, u10_era5, u10_era5_stddev, u10_ecmwf, u10_ecmwf_stddev = get_u10_reanalysis(args.fid, args.plume_lat, args.plume_lon)
    
    print(u10_hrrr, u10_hrrr_stddev, u10_era5, u10_era5_stddev, u10_ecmwf, u10_ecmwf_stddev)


def herbie_hrrr(plume_lat, plume_lon, date, hour_rounded): 
    """
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    date: str year-month-day e.g., "2024-03-05"
    hour_rounded: e.g., 19:00:00
    
    u10_avg: average 10 m windspeed over 3 x 3 pixel square [m/s]
    u10_std: standard deviation 10 m windspeed over 3 x 3 pixel square [m/s]
    """
    
    ## Get HRRR data as xarray 
    H = Herbie(
    date + ' ' + hour_rounded,  # model run date/time
    model="hrrr",  # model name
    fxx=0,  # forecast lead time
    )

    # Subset xarray to U and V wind at 10-m above ground
    ds = H.xarray(":UGRD:10 m")
    
    # Find closest point in HRRR grid to desired lat/lon
    sub_array = [ds[var].values for var in ['latitude', 'longitude']]
    abs_diffs = [np.abs(arr - target) for arr, target in zip(sub_array, [plume_lat,plume_lon])]
    total_diff = sum(abs_diffs)
    min_index = np.unravel_index(np.argmin(total_diff), total_diff.shape)
    
    # Extract U10 windspeed at 3x3 grid around index
    u10_data = ds['u10']
    x_min, x_max = max(min_index[0] - 1, 0), min(min_index[0] + 2, u10_data.shape[0])
    y_min, y_max = max(min_index[1] - 1, 0), min(min_index[1] + 2, u10_data.shape[1])
    u10_data_3x3 = u10_data.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    
    # Standard dev and mean of windspeed calc
    u10_avg = np.abs(np.nanmean(u10_data_3x3))
    u10_std = np.abs(np.nanstd(u10_data_3x3))

    return u10_avg, u10_std

def open_meteo_era5(plume_lat, plume_lon, date, hour_rounded, model, grid_size = 0.1): 
    """
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    date: str year-month-day e.g., "2024-03-05"
    hour_rounded: e.g., 19:00:00
    model: reanalsyis models on open-meteo ("ecmwf_ifs", "best_match", "era5_seamless")
    grid_size: degree value to average over (0.1 = 0.3 x 0.3 deg square) 

    u10_avg: average 10 m windspeed over 3 x grid_size square [m/s]
    u10_std: standard deviation 10 m windspeed over 3 x grid_size square [m/s]
    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Call Open-Meteo API
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": [plume_lat-grid_size, plume_lat, plume_lat+grid_size, 
                plume_lat-grid_size, plume_lat, plume_lat+grid_size, 
                plume_lat-grid_size, plume_lat, plume_lat+grid_size],
        "longitude": [plume_lon+grid_size, plume_lon+grid_size, plume_lon+grid_size,
                  plume_lon, plume_lon, plume_lon, 
                  plume_lon-grid_size, plume_lon-grid_size, plume_lon-grid_size],
        "start_date": date,
        "end_date": date,
        "hourly": "wind_speed_10m",
        "models": model, 
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

def get_cluster_paths(fid): 
    """
    fid: EMIT/AV3 file ID
    
    loc_path: path to orthorectified LOC file
    obs_path: path to orthorectified OBS file
    """
    if fid[0:4] == 'emit': 
        base_path = '/store/emit/ops/data/acquisitions/'
        sub_path = os.path.join(base_path, fid[4:12], fid, 'l1b')
        obs_path = glob.glob(os.path.join(sub_path, fid + '*obs*.hdr'))[0]
        loc_path = glob.glob(os.path.join(sub_path, fid +'*loc*.hdr'))[0]
        
    elif fid[0:3] == 'AV3': 
        try: 
            obs_path = glob.glob(os.path.join('/store/av3/y24/rdn/ort/', fid +'*OBS_ORT*.hdr'))[0] 
            loc_path = glob.glob(os.path.join('/store/av3/y24/rdn/ort/', fid +'*LOC_ORT*.hdr'))[0]
        except: 
            obs_path = glob.glob(os.path.join('/store/av3/y23/rdn/ort/', fid +'*OBS_ORT*.hdr'))[0] 
            loc_path = glob.glob(os.path.join('/store/av3/y23/rdn/ort/', fid +'*LOC_ORT*.hdr'))[0]
    return obs_path, loc_path

def get_u10_reanalysis(fid, plume_lat, plume_lon): 
    """
    fid: EMIT/AV3 file ID
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    
    u10_hrrr/era5/ecmwf: averaged u10 reanalysis product
    u10_hrrr/era5/ecmwf_stddev: spatial standard dev. of u10 reanalysis product
    """
    
    obs_path, loc_path = get_cluster_paths(fid)
    
    # Find row/col closest to given lat/lon
    loc_ds = envi.open(envi_header(loc_path)).open_memmap(interleave='bil')[:,0:2,:]
    lon_cond = np.abs(loc_ds[:,0,:] - plume_lon)
    lat_cond = np.abs(loc_ds[:,1,:] - plume_lat)
    matched = np.unravel_index(np.argmin(lon_cond + lat_cond), lon_cond.shape)  
    
    # Get exact overpass timing for given lat/lon
    obs_ds = envi.open(envi_header(os.path.join(obs_path))).open_memmap(interleave='bip')
    frac_time = obs_ds[matched[0],matched[1],9]
    acq_time_pre = int(np.floor(frac_time))
    acq_time_post = int(np.ceil(frac_time))
    
    # TODO: Check to see if pre/post time puts you into different day
    hour_rounded_pre = str(acq_time_pre) + ':00:00'
    hour_rounded_post = str(acq_time_post) + ':00:00'
    if fid[0:4] == 'emit': 
        date = fid[4:8] + '-' + fid[8:10] + '-' + fid[10:12]
    elif fid[0:3] == 'AV3':
        date = fid[3:7] + '-' + fid[7:9] + '-' + fid[9:11]
            
    # Check if source is in HRRR bounds
    if (21.13812300000003 <= plume_lat <= 52.61565330680793) and (225.90452026573686 <= plume_lon%360 <= 299.0828072281622): 
        u10_avg_pre, u10_std_pre = herbie_hrrr(plume_lat, plume_lon%360, date, hour_rounded_pre)
        u10_avg_post, u10_std_post = herbie_hrrr(plume_lat, plume_lon%360, date, hour_rounded_post)
        u10_hrrr= np.interp(frac_time, [acq_time_pre, acq_time_post], [u10_avg_pre, u10_avg_post])
        u10_hrrr_stddev = np.interp(frac_time, [acq_time_pre, acq_time_post], [u10_std_pre, u10_std_post])
    else: 
        u10_hrrr, u10_hrrr_stddev = np.nan, np.nan
        
    # ERA-5
    u10_avg_pre, u10_std_pre = open_meteo_era5(plume_lat, plume_lon, date, hour_rounded_pre, model = "era5_seamless")
    u10_avg_post, u10_std_post = open_meteo_era5(plume_lat, plume_lon, date, hour_rounded_post, model = "era5_seamless") 
    u10_era5 = np.interp(frac_time, [acq_time_pre, acq_time_post], [u10_avg_pre, u10_avg_post])
    u10_era5_stddev = np.interp(frac_time, [acq_time_pre, acq_time_post], [u10_std_pre, u10_std_post])
    
    # ECMWF
    u10_avg_pre, u10_std_pre = open_meteo_era5(plume_lat, plume_lon, date, hour_rounded_pre, model = "ecmwf_ifs")
    u10_avg_post, u10_std_post = open_meteo_era5(plume_lat, plume_lon, date, hour_rounded_post, model = "ecmwf_ifs") 
    u10_ecmwf = np.interp(frac_time, [acq_time_pre, acq_time_post], [u10_avg_pre, u10_avg_post])
    u10_ecmwf_stddev = np.interp(frac_time, [acq_time_pre, acq_time_post], [u10_std_pre, u10_std_post])
    
    return u10_hrrr, u10_hrrr_stddev, u10_era5, u10_era5_stddev, u10_ecmwf, u10_ecmwf_stddev

if __name__ == '__main__':
    main()

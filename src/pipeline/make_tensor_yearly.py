# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:07:33 2026

@author: jdanielou
"""
import xarray as xr
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import glob
import os
import time
from datetime import datetime

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def create_buffered_extent(extent, buffer_deg=1.0):
    return {
        'lat_min': extent['lat_min'] - buffer_deg,
        'lat_max': extent['lat_max'] + buffer_deg,
        'lon_min': extent['lon_min'] - buffer_deg,
        'lon_max': extent['lon_max'] + buffer_deg
    }

def get_target_files(nc_dir, time_bounds):
    dates = pd.date_range(start=time_bounds['start'], end=time_bounds['end'])
    date_strings = [f"_{d.strftime('%Y%m%d')}_" for d in dates] 
    
    all_files = glob.glob(os.path.join(nc_dir, "*.nc"))
    return [f for f in all_files if any(ds in f for ds in date_strings)]

def get_gfw_target_files(csv_pattern, time_bounds):
    dates = pd.date_range(start=time_bounds['start'], end=time_bounds['end'])
    date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    date_strings_alt = [d.strftime('%Y%m%d') for d in dates]
    
    all_files = glob.glob(csv_pattern)
    return [
        f for f in all_files 
        if any(ds in f for ds in date_strings) or any(ds in f for ds in date_strings_alt)
    ]

def load_and_rasterize_gfw(csv_pattern, extent, time_bounds):
    target_files = get_gfw_target_files(csv_pattern, time_bounds)
    log(f"GFW : Chargement de {len(target_files)} fichiers")
    
    if len(target_files) == 0:
        raise ValueError("Erreur : Aucun fichier GFW ne correspond a la periode demandee.")
    
    gfw_dtypes = {
        'fishing_hours': 'float64',
        'hours': 'float64',
        'mmsi_present': 'float64',
        'cell_ll_lat': 'float64',
        'cell_ll_lon': 'float64',
        'geartype': 'object',
        'date': 'object'
    }
    
    df = dd.read_csv(target_files, dtype=gfw_dtypes)
    
    df['cell_ll_lon'] = df['cell_ll_lon'] % 360
    
    df = df[(df['geartype'] == 'squid_jigger') & 
            (df['cell_ll_lat'] >= extent['lat_min']) & (df['cell_ll_lat'] <= extent['lat_max']) &
            (df['cell_ll_lon'] >= extent['lon_min']) & (df['cell_ll_lon'] <= extent['lon_max'])]
    
    res = 1 / 12.0
    df['lat_bin'] = (df['cell_ll_lat'] / res).astype(int) * res
    df['lon_bin'] = (df['cell_ll_lon'] / res).astype(int) * res
    
    grouped = df.groupby(['date', 'lat_bin', 'lon_bin']).agg({
        'fishing_hours': 'sum',
        'hours': 'sum',
        'mmsi_present': 'sum'
    })
    
    log("GFW : Execution du graphe Dask")
    with ProgressBar():
        grouped = grouped.compute()
    
    grouped['confidence_index'] = (grouped['fishing_hours'] / grouped['hours']) * grouped['mmsi_present']
    grouped = grouped.drop(columns=['hours', 'mmsi_present'])
    grouped.index.names = ['time', 'latitude', 'longitude']
    
    log("GFW : Conversion matricielle")
    grouped = grouped.astype(np.float32)
    
    global_lats = np.unique(grouped.index.get_level_values('latitude'))
    global_lons = np.unique(grouped.index.get_level_values('longitude'))
    
    times = pd.to_datetime(grouped.index.get_level_values('time'))
    years = times.year
    
    ds_list = []
    for y in np.unique(years):
        df_y = grouped[years == y]
        ds_y = df_y.to_xarray()
        
        ds_y = ds_y.reindex(latitude=global_lats, longitude=global_lons, fill_value=0)
        ds_y = ds_y.chunk({'time': -1, 'latitude': 150, 'longitude': 150})
        ds_list.append(ds_y)
        
    ds_gfw = xr.concat(ds_list, dim='time')
    ds_gfw = ds_gfw.assign_coords(time=pd.to_datetime(ds_gfw.time.values))
    ds_gfw = ds_gfw.sortby('time')
    ds_gfw['time'] = ds_gfw.time.dt.floor('D')
    
    return ds_gfw

def load_ocean_data(nc_dir, extent, time_bounds, variables=None):
    target_files = get_target_files(nc_dir, time_bounds)
    log(f"Ocean : Chargement de {len(target_files)} fichiers")
    
    t0 = time.time()
    ds = xr.open_mfdataset(target_files, combine='by_coords', parallel=False)
    
    ds = ds.assign_coords(longitude=(ds.longitude % 360))
    ds = ds.sortby('longitude')
    
    ds = ds.sel(
        latitude=slice(extent['lat_min'], extent['lat_max']),
        longitude=slice(extent['lon_min'], extent['lon_max'])
    )
    
    if 'depth' in ds.dims:
        ds = ds.sel(depth=0, method='nearest')
        
    if variables is not None:
        ds = ds[variables]
        
    ds['time'] = ds.time.dt.floor('D')
    log(f"Ocean : Metadonnees traitees en {time.time() - t0:.2f} secondes")
    return ds

def engineer_features(ds_glorys):
    log("Variables : Calcul des gradients et EKE")
    ds_glorys['sst_grad'] = np.sqrt(ds_glorys['thetao'].differentiate('longitude')**2 + 
                                   ds_glorys['thetao'].differentiate('latitude')**2)
    ds_glorys['eke'] = 0.5 * (ds_glorys['uo']**2 + ds_glorys['vo']**2)
    ds_glorys['so_grad'] = np.sqrt(ds_glorys['so'].differentiate('longitude')**2 + 
                                  ds_glorys['so'].differentiate('latitude')**2)
    return ds_glorys

def process_gebco(bathy_path, ds_reference):
    log("Bathymetrie : Traitement GEBCO")
    ds_bathy = xr.open_dataset(bathy_path)
    
    if 'lat' in ds_bathy.dims:
        ds_bathy = ds_bathy.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    ds_bathy = ds_bathy.assign_coords(longitude=(ds_bathy.longitude % 360))
    ds_bathy = ds_bathy.sortby('longitude')
    
    ds_bathy['elevation'] = ds_bathy['elevation'].where(ds_bathy['elevation'] <= 0)
    
    return ds_bathy.interp_like(ds_reference, method='nearest')

if __name__ == "__main__":
    global_t0 = time.time()
    log("--- Demarrage du pipeline ETL (Mode Production Final) ---")
    
    target_extent = {'lat_min': -57, 'lat_max': 3, 'lon_min': 149, 'lon_max': 291}
    buffered_extent = create_buffered_extent(target_extent, buffer_deg=1.0)
    
    time_bounds_full = {'start': '2012-01-01', 'end': '2024-12-31'}
    path_gfw_csv = "D:/data/fleet-daily/raw/*.csv"
    
    ds_gfw_full = load_and_rasterize_gfw(path_gfw_csv, buffered_extent, time_bounds_full)
    
    glorys_vars = ['thetao', 'so', 'uo', 'vo', 'zos', 'mlotst']
    path_glorys_dir = "U:/glorys_data/daily"
    path_gebco_nc = "D:/data/GEBCO/gebco_2026_n4.0_s-58.0_w148.0_e292.0.nc"
    path_output_dir = "D:/deep-learning-squid-prediction/data/tensors_annuels/v2"
    os.makedirs(path_output_dir, exist_ok=True)
    
    log("Initialisation du referentiel spatial global...")
    time_bounds_ref = {'start': '2012-01-01', 'end': '2012-01-01'}
    ds_ref = load_ocean_data(path_glorys_dir, target_extent, time_bounds_ref, variables=glorys_vars)
    ds_bathy_global = process_gebco(path_gebco_nc, ds_ref)
    ds_ref.close()
    
    years = range(2012, 2025)
    
    for y in years:
        log(f" Traitement strict et isolation de l'annee {y}")
        t_year = time.time()
        
        try:
            time_bounds_y = {'start': f'{y}-01-01', 'end': f'{y}-12-31'}
            
            ds_glorys_y = load_ocean_data(path_glorys_dir, target_extent, time_bounds_y, variables=glorys_vars)
            ds_ocean_y = engineer_features(ds_glorys_y)
            
            ds_gfw_y = ds_gfw_full.sel(time=str(y))
            ds_gfw_y = ds_gfw_y.reindex(latitude=ds_ocean_y.latitude, longitude=ds_ocean_y.longitude, method='nearest')
            ds_gfw_y = ds_gfw_y.reindex(time=ds_ocean_y.time)
            
            ds_partial_y = xr.merge([ds_ocean_y, ds_bathy_global, ds_gfw_y], join='exact')
            
            ds_partial_y['fishing_hours'] = ds_partial_y['fishing_hours'].where(ds_partial_y['elevation'] < 0, 0)
            ds_partial_y['confidence_index'] = ds_partial_y['confidence_index'].where(ds_partial_y['elevation'] < 0, 0)
            
            ds_partial_y = ds_partial_y.chunk({'time': 30, 'latitude': 300, 'longitude': 300})
            ds_partial_y = ds_partial_y.fillna(0)
            
            if 'crs' in ds_partial_y.data_vars or 'crs' in ds_partial_y.coords:
                ds_partial_y = ds_partial_y.drop_vars('crs')
                
            ds_partial_y = ds_partial_y.astype('float32')
            
            compression = dict(zlib=True, complevel=5)
            encoding = {var: compression for var in ds_partial_y.data_vars}
            path_output_nc = os.path.join(path_output_dir, f"tensor_{y}.nc")
            
            log(f"Ecriture sur le disque pour {y}...")
            with ProgressBar():
                ds_partial_y.to_netcdf(path_output_nc, encoding=encoding)
            
            ds_partial_y.close()
            log(f"Annee {y} terminee en {(time.time() - t_year) / 60:.2f} minutes.")
            
        except Exception as e:
            log(f"Erreur critique sur l'annee {y} : {e}")
            continue 
            
    log(f"--- Pipeline global termine en {(time.time() - global_t0) / 3600:.2f} heures ---")
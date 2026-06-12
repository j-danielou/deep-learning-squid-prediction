# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:35:42 2026

@author: jdanielou
"""
import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import time
from datetime import datetime

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_target_files(nc_dir, year):
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
    date_strs_1 = [d.strftime('%Y%m%d') for d in dates]
    date_strs_2 = [d.strftime('%Y-%m-%d') for d in dates]
    
    all_files = glob.glob(os.path.join(nc_dir, "*.nc"))
    return [f for f in all_files if any(ds in f for ds in date_strs_1) or any(ds in f for ds in date_strs_2)]

def process_chlorophyll_yearly(path_chla_dir, path_tensors_dir):
    log("Demarrage du pipeline Chlorophylle - Version Vectorisee")
    
    years = range(2012, 2025)
    
    for y in years:
        log(f"Traitement de l'annee {y} ")
        t_year = time.time()
        
        ref_path = os.path.join(path_tensors_dir, f"tensor_{y}.nc")
        if not os.path.exists(ref_path):
            log(f"Fichier de reference introuvable pour {y}. Ignore.")
            continue
            
        #Grid ref
        with xr.open_dataset(ref_path) as ds_ref:
            ref_lat = ds_ref.latitude
            ref_lon = ds_ref.longitude
        
        target_files = get_target_files(path_chla_dir, y)
        if not target_files:
            log(f"Aucun fichier CHL trouve pour {y}. Ignore.")
            continue
            
        log("Ouverture et alignement du dataset.")
        ds_y = xr.open_mfdataset(target_files, combine='by_coords', parallel=True, chunks={'time': 30})
        
        rename_dict = {}
        if 'lon' in ds_y.coords or 'lon' in ds_y.dims:
            rename_dict['lon'] = 'longitude'
        if 'lat' in ds_y.coords or 'lat' in ds_y.dims:
            rename_dict['lat'] = 'latitude'
        if rename_dict:
            ds_y = ds_y.rename(rename_dict)
        
        ds_y = ds_y.assign_coords(longitude=(ds_y.longitude % 360))
        ds_y = ds_y.sortby(['longitude', 'latitude'])
        
        #filtrage spatial
        ds_y = ds_y.sel(
            latitude=slice(ref_lat.min() - 0.1, ref_lat.max() + 0.1),
            longitude=slice(ref_lon.min() - 0.1, ref_lon.max() + 0.1)
        )
        
        #extraction features
        ds_chl = ds_y[['CHL']]
        
        log("Interpolation spatiale globale en cours.")
        t_math = time.time()
        
        #Interpolation
        ds_interp = ds_chl.interp(
            latitude=ref_lat,
            longitude=ref_lon,
            method='linear'
        )
        
        #Nettoyage
        ds_interp = ds_interp.fillna(0).astype('float32')
        
        #Calcul Matriciel
        ds_interp['chl_grad'] = np.sqrt(
            ds_interp['CHL'].differentiate('longitude')**2 + 
            ds_interp['CHL'].differentiate('latitude')**2
        ).astype('float32')
        
        log(f"Operations mathematiques terminees en {time.time() - t_math:.2f} secondes.")
        
        ds_interp = ds_interp.chunk({'time': 30, 'latitude': 300, 'longitude': 300})
        
        output_path = os.path.join(path_tensors_dir, f"tensor_chl_{y}.nc")
        compression = dict(zlib=True, complevel=5)
        encoding = {'CHL': compression, 'chl_grad': compression}
        
        log("Evaluation du graphe Dask et ecriture sur disque")
        t_write = time.time()
        ds_interp.to_netcdf(output_path, encoding=encoding)
        log(f"Ecriture terminee en {time.time() - t_write:.2f} secondes.")
        
        ds_y.close()
        ds_interp.close()
        
        log(f"Annee {y} terminee en {(time.time() - t_year) / 60:.2f} minutes.\n")

if __name__ == "__main__":
    path_chla_daily = "U:/glorys_data/chla_daily/2012-2024"
    path_tensors_annuels = "D:/deep-learning-squid-prediction/data/tensors_annuels/v2"
    
    process_chlorophyll_yearly(path_chla_daily, path_tensors_annuels)
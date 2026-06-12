# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:04:50 2026

@author: jdanielou
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os

def audit_and_plot_tensors(tensor_dir):
    print("Demarrage de l'audit des tenseurs de chlorophylle...\n")
    
    search_pattern = os.path.join(tensor_dir, "tensor_chl_*.nc")
    files = glob.glob(search_pattern)
    files.sort()
    
    if not files:
        print("Aucun fichier trouve dans le repertoire cible.")
        return
        
    print(f"Nombre de fichiers trouves : {len(files)}\n")
    
    for f in files:
        filename = os.path.basename(f)
        try:
            with xr.open_dataset(f) as ds:
                #Verif dims
                time_len = ds.sizes.get('time', 0)
                lat_len = ds.sizes.get('latitude', 0)
                lon_len = ds.sizes.get('longitude', 0)
                
                #Extraction
                sample_chl = ds['CHL'].isel(time=0).values
                sample_grad = ds['chl_grad'].isel(time=0).values
                
                
                print(f"Fichier : {filename}")
                print(f"  Shape : (Time: {time_len}, Lat: {lat_len}, Lon: {lon_len})")
                print(f"  [CHL]      NaN: {np.isnan(sample_chl).any():<5} | Inf: {np.isinf(sample_chl).any():<5} | Min: {np.nanmin(sample_chl):.4f} | Max: {np.nanmax(sample_chl):.4f}")
                print(f"  [Gradient] NaN: {np.isnan(sample_grad).any():<5} | Inf: {np.isinf(sample_grad).any():<5} | Min: {np.nanmin(sample_grad):.4f} | Max: {np.nanmax(sample_grad):.4f}\n")
                
        except Exception as e:
            print(f"Erreur lors de la lecture de {filename} : {str(e)}\n")
            
    #
    print("Generation de la carte de controle")
    first_file = files[0]
    
    ds = xr.open_dataset(first_file)
    day1 = ds.isel(time=0)
    
    fig = plt.figure(figsize=(18, 8))
    proj = ccrs.PlateCarree(central_longitude=180.0)
    
    #Plot CHL
    ax1 = fig.add_subplot(1, 2, 1, projection=proj)
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    mesh1 = ax1.pcolormesh(
        day1.longitude, 
        day1.latitude, 
        day1.CHL,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0.0,
        vmax=2.0, 
        zorder=1
    )
    ax1.set_title(f"Chlorophylle ({str(day1.time.values)[:10]})")
    plt.colorbar(mesh1, ax=ax1, shrink=0.6, pad=0.04, label="CHL (mg/m3)")
    
    #Plot grad_CHL
    ax2 = fig.add_subplot(1, 2, 2, projection=proj)
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    mesh2 = ax2.pcolormesh(
        day1.longitude, 
        day1.latitude, 
        day1.chl_grad,
        transform=ccrs.PlateCarree(),
        cmap='magma', 
        vmin=0.0,
        vmax=0.5,
        zorder=1
    )
    ax2.set_title("Gradient Spatial de Chlorophylle")
    plt.colorbar(mesh2, ax=ax2, shrink=0.6, pad=0.04, label="Norme du Gradient")
    
    plt.tight_layout()
    plt.show()
    
    ds.close()

if __name__ == "__main__":
    path_tensors = "D:/deep-learning-squid-prediction/data/tensors_annuels/v2"
    audit_and_plot_tensors(path_tensors)
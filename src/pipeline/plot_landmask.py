# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:10:40 2026

@author: jdanielou
"""
import xarray as xr
import matplotlib.pyplot as plt

def plot_glorys_land_mask(nc_path):
    print(f"Chargement du tenseur : {nc_path}")
    ds = xr.open_dataset(nc_path)
    
    salinity_day1 = ds['so'].isel(time=0)
    
    land_mask = (salinity_day1 == 0).astype(int)
    
    print("Génération de la carte du masque...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    land_mask.plot(
        ax=ax, 
        cmap='cividis', 
        add_colorbar=False
    )
    
    ax.set_title("Masque Terrestre Natif GLORYS (Salinité == 0)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    ax.text(0.05, 0.95, 'Jaune : Terre (Salinité = 0)\nBleu foncé : Océan (Salinité > 0)', 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    ds.close()

if __name__ == "__main__":
    path_nc = "D:/deep-learning-squid-prediction/data/tensors_annuels/v2/tensor_2012.nc"
    plot_glorys_land_mask(path_nc)
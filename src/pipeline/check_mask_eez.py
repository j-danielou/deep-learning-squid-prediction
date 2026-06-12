# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:34:48 2026

@author: jdanielou
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

def verify_eez_mask(mask_path):
    print("Ouverture du masque")
    ds = xr.open_dataset(mask_path)
    mask_da = ds['ocean_mask']
    
    print(f"Dimensions du tenseur : Lat {mask_da.sizes.get('latitude')}, Lon {mask_da.sizes.get('longitude')}")
    
    valeurs = mask_da.values
    
    has_nan = np.isnan(valeurs).any()
    print(f"Présence de NaN ? : {has_nan}")
    
    valeurs_uniques = np.unique(valeurs[~np.isnan(valeurs)])
    print(f"Valeurs uniques présentes : {valeurs_uniques}")
    
    if len(valeurs_uniques) == 2 and 0.0 in valeurs_uniques and 1.0 in valeurs_uniques:
        print("✅ Parfait : Le masque est strictement binaire (0 ou 1).")
    else:
        print("⚠️ Attention : Le masque contient d'autres valeurs que 0 et 1 !")

    print("Génération de la carte")
    
    fig = plt.figure(figsize=(12, 8))
    # Centrage sur le Pacifique
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180.0))
    
    ax.add_feature(cfeature.LAND, facecolor='black', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, color='white', zorder=4)
    
    cmap_binaire = ListedColormap(['crimson', 'dodgerblue'])
    
    mesh = ax.pcolormesh(
        mask_da.longitude, 
        mask_da.latitude, 
        mask_da,
        transform=ccrs.PlateCarree(),
        cmap=cmap_binaire,
        zorder=1
    )
    
    cbar = plt.colorbar(mesh, ax=ax, ticks=[0.25, 0.75], shrink=0.5, pad=0.05)
    cbar.ax.set_yticklabels(['0 : ZEE / Ignoré', '1 : Haute Mer / Appris'])
    
    ax.set_title("Validation du Masque ZEE - Pacifique Est", fontsize=14, pad=15)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.show()
    
    ds.close()

if __name__ == "__main__":
    CHEMIN_MASQUE = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
    verify_eez_mask(CHEMIN_MASQUE)
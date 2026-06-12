# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:49:35 2026

@author: jdanielou
"""
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

def audit_concordance_mask_catch(csv_path, mask_path):
    print("Chargement et preparation des donnees de capture")
    df_raw = pd.read_csv(csv_path)
    df_gigas = df_raw[df_raw["species_code"] == "GIS"].copy()
    
    df_gigas["harvest_kg"] = pd.to_numeric(df_gigas["harvest_kg"], errors="coerce")
    df_gigas["lat"] = pd.to_numeric(df_gigas["lat"], errors="coerce")
    df_gigas["long"] = pd.to_numeric(df_gigas["long"], errors="coerce")
    df_gigas = df_gigas.dropna(subset=["harvest_kg", "lat", "long"])
    
    df_gigas["lon_360"] = df_gigas["long"] % 360
    
    print("Chargement du masque officiel ZEE")
    ds_mask = xr.open_dataset(mask_path)
    mask_da = ds_mask['ocean_mask']
    
    lat_min, lat_max = mask_da.latitude.min().item(), mask_da.latitude.max().item()
    lon_min, lon_max = mask_da.longitude.min().item(), mask_da.longitude.max().item()
    
    mask_bounds = (df_gigas["lat"] >= lat_min) & (df_gigas["lat"] <= lat_max) & \
                  (df_gigas["lon_360"] >= lon_min) & (df_gigas["lon_360"] <= lon_max)
    
    df_valid = df_gigas[mask_bounds].copy()
    
    print("Extraction vectorisee des statuts geographiques")
    lats_xr = xr.DataArray(df_valid["lat"].values, dims="points")
    lons_xr = xr.DataArray(df_valid["lon_360"].values, dims="points")
    
    extracted_status = mask_da.sel(latitude=lats_xr, longitude=lons_xr, method="nearest").values
    df_valid["is_high_seas"] = extracted_status
    
    total_points = len(df_valid)
    retained_points = len(df_valid[df_valid["is_high_seas"] == 1.0])
    excluded_points = len(df_valid[df_valid["is_high_seas"] == 0.0])
    
    retention_rate = (retained_points / total_points) * 100
    
    print("\nRAPPORT")
    print(f"Total des points de capture evalues : {total_points}")
    print(f"Points conserves pour l'apprentissage (Haute Mer) : {retained_points}")
    print(f"Points exclus/masques par la Loss (Dans la ZEE)   : {excluded_points}")
    print(f"Taux de retention des donnees utiles              : {retention_rate:.2f} %")
    
    print("Generation de la cartographie de controle")
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180.0))
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)

    cmap_bg = ListedColormap(['#ffcccc', '#ccedff']) 
    ax.pcolormesh(
        mask_da.longitude, 
        mask_da.latitude, 
        mask_da,
        transform=ccrs.PlateCarree(),
        cmap=cmap_bg,
        alpha=0.5,
        zorder=1
    )
    
    df_excluded = df_valid[df_valid["is_high_seas"] == 0.0]
    ax.scatter(
        df_excluded["long"], 
        df_excluded["lat"],
        color='red',
        s=15,
        edgecolor='black',
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label=f'Exclus (ZEE) : {len(df_excluded)}'
    )
    
    df_retained = df_valid[df_valid["is_high_seas"] == 1.0]
    ax.scatter(
        df_retained["long"],
        df_retained["lat"],
        color='navy',
        s=15,
        edgecolor='white',
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label=f'Conserves (Haute Mer) : {len(df_retained)}'
    )
    
    ax.set_title("Audit Spatial : Captures retenues vs exclues par le Masque ZEE", fontsize=14, pad=15)
    ax.legend(loc='lower left', framealpha=0.9)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    plt.tight_layout()
    plt.show()
    
    ds_mask.close()

if __name__ == "__main__":
    CHEMIN_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
    CHEMIN_MASQUE = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
    
    audit_concordance_mask_catch(CHEMIN_CSV, CHEMIN_MASQUE)
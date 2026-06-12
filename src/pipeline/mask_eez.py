# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:02:09 2026

@author: jdanielou
"""

# -*- coding: utf-8 -*-
"""
Création du Masque ZEE Officiel (Rasterization)
À exécuter une seule fois pour générer le masque de référence.
"""
import xarray as xr
import geopandas as gpd
import regionmask
import numpy as np

def generate_official_eez_mask(shapefile_path, ref_nc_path, output_path):
    print("1. Chargement de la grille environnementale de référence")
    ds_ref = xr.open_dataset(ref_nc_path)
    
    ds_ref = ds_ref.sel(
        latitude=slice(-38.0, 3.0),
        longitude=slice(227.0, 293.0)
    )
    
    print("2. Chargement des frontières ZEE (MarineRegions)")
    eez_gdf = gpd.read_file(shapefile_path)
    
    eez_gdf = eez_gdf.dropna(subset=['geometry'])
    eez_gdf = eez_gdf[eez_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    print("3. Rasterization : 'Brûlage' des polygones sur la grille océanique")
    eez_regions = regionmask.from_geopandas(eez_gdf, overlap=False)
    
    mask_2d = eez_regions.mask(ds_ref.longitude, ds_ref.latitude)
    
    print("4. Finalisation mathématique du masque PyTorch")
    high_seas_mask = np.isnan(mask_2d).astype(np.float32)
    
    ref_chl = ds_ref['so'].isel(time=0).values
    land_mask = (ref_chl != 0.0) & (~np.isnan(ref_chl))
    
    final_mask = high_seas_mask * land_mask
    
    print("5. Sauvegarde du masque au format NetCDF")
    ds_mask = xr.Dataset(
        {"ocean_mask": (["latitude", "longitude"], final_mask.values)},
        coords={
            "latitude": ds_ref.latitude,
            "longitude": ds_ref.longitude,
        }
    )
    
    ds_mask.to_netcdf(output_path)
    ds_ref.close()
    print(f"✅ Terminé ! Masque sauvegardé ici : {output_path}")

if __name__ == "__main__":
    PATH_SHAPEFILE = "D:/data/World_EEZ_v12_20231025_0_360/eez_v12_0_360.shp" 
    PATH_TENSOR_REF = "D:/deep-learning-squid-prediction/data/tensors_annuels/tensor_2012.nc"
    PATH_OUTPUT = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
    
    generate_official_eez_mask(PATH_SHAPEFILE, PATH_TENSOR_REF, PATH_OUTPUT)
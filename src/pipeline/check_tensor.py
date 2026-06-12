# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:26:10 2026

@author: jdanielou
"""

import xarray as xr
import numpy as np

def run_tensor_diagnostics(file_path, expected_year=2012):
    print("--- Demarrage de l'audit complet du tenseur ---")
    
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"Erreur fatale : Impossible d'ouvrir le fichier NetCDF.\n{e}")
        return

    print("\n1. Verification Structurelle")
    expected_vars = [
        'thetao', 'so', 'uo', 'vo', 'zos', 'mlotst', 
        'sst_grad', 'eke', 'so_grad', 'elevation', 
        'fishing_hours', 'confidence_index'
    ]
    
    missing_vars = [v for v in expected_vars if v not in ds.data_vars]
    if missing_vars:
        print(f"ALERTE : Variables manquantes : {missing_vars}")
    else:
        print("OK : Toutes les variables attendues sont presentes.")

    is_leap = (expected_year % 4 == 0 and expected_year % 100 != 0) or (expected_year % 400 == 0)
    expected_days = 366 if is_leap else 365
    
    if ds.sizes['time'] != expected_days:
        print(f"ALERTE : Dimension temporelle incorrecte. Attendu: {expected_days}, Obtenu: {ds.sizes['time']}")
    else:
        print(f"OK : Dimension temporelle valide ({expected_days} jours).")

    print("\n2. Verification de l'Integrite Numerique (NaN et Inf)")
    for var in ds.data_vars:
        has_nan = ds[var].isnull().any().item()
        has_inf = np.isinf(ds[var]).any().item()
        
        if has_nan or has_inf:
            print(f"ALERTE : Valeurs corrompues detectees dans {var} (NaN: {has_nan}, Inf: {has_inf})")
        else:
            print(f"OK : Proprete numerique validee pour {var}")

    print("\n3. Verification de la Coherence Physique (Sanity Check)")
    
    temp_min, temp_max = ds['thetao'].min().item(), ds['thetao'].max().item()
    if temp_min < -5 or temp_max > 40:
        print(f"ALERTE : Temperatures aberrantes (Min: {temp_min:.2f}, Max: {temp_max:.2f})")
    else:
        print(f"OK : Plage de temperature coherente ({temp_min:.2f} a {temp_max:.2f} C).")

    bathy_max = ds['elevation'].max().item()
    if bathy_max > 0:
        print(f"ALERTE : La bathymetrie contient des elevations terrestres positives ({bathy_max} m).")
    else:
        print("OK : Le masque bathymetrique est strictement marin/côtier (<= 0).")

    fishing_min = ds['fishing_hours'].min().item()
    if fishing_min < 0:
        print(f"ALERTE : Heures de peche negatives detectees ({fishing_min}).")
    else:
        print("OK : L'effort de peche est strictement positif ou nul.")

    print("\n4. Verification de l'Alignement Spatial")
    
    land_mask = ds['elevation'] == 0
    fishing_on_land = ds['fishing_hours'].where(land_mask).max().item()
    
    if fishing_on_land > 0:
        print(f"ALERTE : Activite de peche detectee sur des pixels terrestres (Max: {fishing_on_land}h).")
    else:
        print("OK : Aucun effort de peche detecte sur la terre ferme.")
        
    ds.close()
    print("\n--- Audit termine ---")

if __name__ == "__main__":
    path_nc_2024 = "D:/deep-learning-squid-prediction/data/tensors_annuels/v2/tensor_2024.nc"
    run_tensor_diagnostics(path_nc_2024, expected_year=2024)
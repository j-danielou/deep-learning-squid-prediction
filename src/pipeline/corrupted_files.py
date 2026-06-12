# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:02:48 2026

@author: jdanielou
"""
import xarray as xr
import glob
import os

def identify_corrupted_files(directory_pattern):
    print("Recherche de fichiers corrompus en cours...")
    files = glob.glob(directory_pattern)
    corrupted_files = []
    
    for file_path in files:
        try:
            #test d'ouverture pour forcer la lecture de la variable time
            ds = xr.open_dataset(file_path, engine='netcdf4')
            ds.time.values 
            ds.close()
        except Exception:
            corrupted_files.append(file_path)
            print(f"Fichier corrompu identifie : {os.path.basename(file_path)}")
            
    if not corrupted_files:
        print("Aucun fichier corrompu detecte.")
    else:
        print(f"Total : {len(corrupted_files)} fichier(s) a supprimer et re-telecharger.")

if __name__ == "__main__":
    target_directory = "U:/glorys_data/chla_daily/2012-2024/*.nc"
    identify_corrupted_files(target_directory)
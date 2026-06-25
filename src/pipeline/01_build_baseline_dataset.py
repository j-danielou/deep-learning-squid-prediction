# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:12:52 2026

@author: jdanielou
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. Configuration
YEARS = list(range(2012, 2025))
PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
DIR_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
DIR_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
DIR_OUT = "D:/deep-learning-squid-prediction/data/dataframes/"

FEATURES_ENV = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad']

def load_global_catch_data(csv_path):
    # Chargement unique de la base SPRFMO pour toutes les annees
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
        
    df_raw = pd.read_csv(csv_path)
    df_gigas = df_raw[df_raw["species_code"] == "GIS"].copy()
    
    df_gigas["harvest_kg"] = pd.to_numeric(df_gigas["harvest_kg"], errors="coerce")
    df_gigas["lat"] = pd.to_numeric(df_gigas["lat"], errors="coerce")
    df_gigas["long"] = pd.to_numeric(df_gigas["long"], errors="coerce")
    df_gigas["year"] = pd.to_numeric(df_gigas["year"], errors="coerce").astype(int)
    df_gigas["month"] = pd.to_numeric(df_gigas["month"], errors="coerce").astype(int)
    
    df_gigas = df_gigas.dropna(subset=["harvest_kg", "lat", "long", "year", "month"])
    df_gigas["lon_360"] = df_gigas["long"] % 360
    df_gigas['time'] = pd.to_datetime(df_gigas[['year', 'month']].assign(day=1))
    
    df_gigas['latitude'] = df_gigas['lat'].round(1)
    df_gigas['longitude'] = df_gigas['lon_360'].round(1)
    df_gigas['capture_tonnes'] = df_gigas['harvest_kg'] / 1000.0
    
    return df_gigas[['time', 'latitude', 'longitude', 'capture_tonnes', 'year']]

def process_single_year(year, df_peche_global):
    out_path = os.path.join(DIR_OUT, f"dataset_ML_baseline_{year}.csv")
    
    # Checkpoint : Si le fichier annuel existe deja, on le saute
    if os.path.exists(out_path):
        return pd.read_csv(out_path)
        
    path_phys = os.path.join(DIR_PHYS, f"tensor_{year}.nc")
    path_chl = os.path.join(DIR_CHL, f"tensor_chl_{year}.nc")
    
    if not (os.path.exists(path_phys) and os.path.exists(path_chl)):
        return None

    ds_phys = xr.open_dataset(path_phys)
    ds_chl = xr.open_dataset(path_chl)
    ds_hr = xr.merge([ds_phys, ds_chl], compat="override")

    da_effort = ds_hr['fishing_hours']
    ds_env = ds_hr.drop_vars('fishing_hours')

    ds_env_mensuel = ds_env.resample(time='1MS').mean()
    da_effort_mensuel = da_effort.resample(time='1MS').sum()

    ds_env_1deg = ds_env_mensuel.coarsen(latitude=12, longitude=12, boundary='trim').mean()
    da_effort_1deg = da_effort_mensuel.coarsen(latitude=12, longitude=12, boundary='trim').sum()

    ds_1deg = xr.merge([ds_env_1deg, da_effort_1deg])

    df_env = ds_1deg.to_dataframe().reset_index()
    df_env['latitude'] = df_env['latitude'].round(1)
    df_env['longitude'] = df_env['longitude'].round(1)
    df_env['time'] = pd.to_datetime(df_env['time'])

    ds_phys.close()
    ds_chl.close()

    df_peche_year = df_peche_global[df_peche_global['year'] == year].drop(columns=['year'])

    df_final = pd.merge(df_env, df_peche_year, on=['time', 'latitude', 'longitude'], how='inner')
    df_final = df_final[df_final['fishing_hours'] > 0].copy()
    
    if df_final.empty:
        return None
        
    df_final['CPUE_tonnes_par_heure'] = df_final['capture_tonnes'] / df_final['fishing_hours']

    colonnes_finales = ['time', 'latitude', 'longitude'] + FEATURES_ENV + ['fishing_hours', 'capture_tonnes', 'CPUE_tonnes_par_heure']
    df_ML = df_final[colonnes_finales].dropna(subset=FEATURES_ENV)
    
    # Sauvegarde du checkpoint annuel
    df_ML.to_csv(out_path, index=False)
    return df_ML

if __name__ == "__main__":
    print("Chargement en RAM de la base SPRFMO (2012-2024)")
    df_peche_global = load_global_catch_data(PATH_CSV)
    print(f"Chargement reussi. Total des evenements GIS : {len(df_peche_global)}")
    
    liste_df_annuels = []
    
    print("\nLancement du traitement par annee...")
    # La barre de progression tqdm entoure l'iterateur
    for year in tqdm(YEARS, desc="Traitement spatio-temporel", unit="annee"):
        df_year = process_single_year(year, df_peche_global)
        if df_year is not None and not df_year.empty:
            liste_df_annuels.append(df_year)
            
    if liste_df_annuels:
        print("\nConcatenation finale des 13 annees...")
        df_complet = pd.concat(liste_df_annuels, ignore_index=True)
        
        chemin_final = os.path.join(DIR_OUT, "dataset_ML_baseline_13_ans.csv")
        df_complet.to_csv(chemin_final, index=False)
        
        print("\n" + "="*50)
        print(" BILAN DU DATASET D'ENTRAINEMENT GLOBAL")
        print("="*50)
        print(f"Annees compilees  : {df_complet['time'].dt.year.nunique()}")
        print(f"Lignes totales    : {len(df_complet)}")
        print(f"CPUE Moyenne      : {df_complet['CPUE_tonnes_par_heure'].mean():.2f} tonnes/h")
        print(f"CPUE Max          : {df_complet['CPUE_tonnes_par_heure'].max():.2f} tonnes/h")
        print("="*50)
        print(f"Dataset global sauvegarde : {chemin_final}")
    else:
        print("\nAucune donnee n'a pu etre compilee.")
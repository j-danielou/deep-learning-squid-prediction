# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:09:51 2026

@author: jdanielou
"""
import pandas as pd
import os

def load_and_prepare_catch_data(csv_path):
    print("Chargement du fichier CSV")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
        
    df_raw = pd.read_csv(csv_path)
    
    print("Filtrage pour l'espece : GIS")
    df_gigas = df_raw[df_raw["species_code"] == "GIS"].copy()
    
    df_gigas["harvest_kg"] = pd.to_numeric(df_gigas["harvest_kg"], errors="coerce")
    df_gigas["lat"] = pd.to_numeric(df_gigas["lat"], errors="coerce")
    df_gigas["long"] = pd.to_numeric(df_gigas["long"], errors="coerce")
    df_gigas["year"] = pd.to_numeric(df_gigas["year"], errors="coerce").astype(int)
    df_gigas["month"] = pd.to_numeric(df_gigas["month"], errors="coerce").astype(int)
    
    df_gigas = df_gigas.dropna(subset=["harvest_kg", "lat", "long", "year", "month"])
    
    df_gigas["lon_360"] = df_gigas["long"] % 360
    df_gigas = df_gigas.reset_index(drop=True)
    
    print(f"Preparation terminee. Evenements valides : {len(df_gigas)}")
    return df_gigas
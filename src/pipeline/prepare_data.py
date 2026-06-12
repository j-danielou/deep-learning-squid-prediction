# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:55:01 2026

@author: jdanielou
"""
import pandas as pd
import os

def load_and_prepare_catch_data(csv_path):
    print("Chargement du fichier CSV")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier est introuvable au chemin : {csv_path}")
        
    df_raw = pd.read_csv(csv_path)
    
    print("Filtrage pour l'espèce : GIS (Dosidicus gigas)")
    df_gigas = df_raw[df_raw["species_code"] == "GIS"].copy()
    
    #Nettoyage
    df_gigas["harvest_kg"] = pd.to_numeric(df_gigas["harvest_kg"], errors="coerce")
    df_gigas["lat"] = pd.to_numeric(df_gigas["lat"], errors="coerce")
    df_gigas["long"] = pd.to_numeric(df_gigas["long"], errors="coerce")
    df_gigas["year"] = pd.to_numeric(df_gigas["year"], errors="coerce").astype(int)
    df_gigas["month"] = pd.to_numeric(df_gigas["month"], errors="coerce").astype(int)
    
    #Suppression des lignes corrompues
    df_gigas = df_gigas.dropna(subset=["harvest_kg", "lat", "long", "year", "month"])
    
    df_gigas["lon_360"] = df_gigas["long"] % 360
    df_gigas = df_gigas.reset_index(drop=True)
    
    print(f"Préparation terminée. Événements valides : {len(df_gigas)}")
    return df_gigas

if __name__ == "__main__":
    file_path = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
    df_catch = load_and_prepare_catch_data(file_path)
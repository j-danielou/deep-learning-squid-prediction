# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:37:32 2026

@author: jdanielou
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
DIR_TENSORS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
SAVE_DIR = "D:/deep-learning-squid-prediction/data/"

MU = 18.0
SIGMA_MONTEE = 5.50
SIGMA_DESCENTE = 5.0
AMPLITUDE = 0.5

R_GROWTH = 1.2
K_CAPACITY = 8.63e10
B_INITIAL = 1.6e9

def calculer_q_matrice(T_matrice):
    mask_inf = T_matrice <= MU
    mask_sup = T_matrice > MU
    
    q = np.zeros_like(T_matrice)
    q[mask_inf] = AMPLITUDE * np.exp(-0.5 * ((T_matrice[mask_inf] - MU) / SIGMA_MONTEE)**2)
    q[mask_sup] = AMPLITUDE * np.exp(-0.5 * ((T_matrice[mask_sup] - MU) / SIGMA_DESCENTE)**2)
    
    return q

def run_simulation_and_evaluate(csv_path, tensors_dir, save_dir):
    print("Chargement des donnees SPRFMO...")
    df = pd.read_csv(csv_path)
    df_gis = df[df['species_code'] == 'GIS'].copy()
    
    years = sorted(df_gis['year'].unique())
    B_t = B_INITIAL
    
    y_true_list = []
    y_pred_list = []
    results_list = []
    
    print("Demarrage de la simulation dynamique")
    
    for year in years:
        tensor_path = os.path.join(tensors_dir, f"tensor_{year}.nc")
        
        if not os.path.exists(tensor_path):
            print(f"Tenseur manquant: {year}. Croissance naturelle appliquee.")
            B_t = B_t + R_GROWTH * B_t * (1 - (B_t / K_CAPACITY))
            continue
            
        df_year = df_gis[df_gis['year'] == year]
        N_events = len(df_year)
        
        if N_events == 0:
            B_t = B_t + R_GROWTH * B_t * (1 - (B_t / K_CAPACITY))
            continue
            
        B_p = B_t / (N_events * 144)
        C_annuel = 0.0
        
        print(f"Annee: {year} | Evenements: {N_events} | Biomasse: {B_t:.2e}")
        
        with xr.open_dataset(tensor_path) as ds:
            for _, row in df_year.iterrows():
                month = int(row['month'])
                lat = row['lat']
                lon = row['long']
                catch_reel = row['harvest_kg']
                
                lon_360 = lon if lon >= 0 else lon + 360
                time_slice = f"{year}-{month:02d}"
                
                try:
                    ds_event = ds.sel(
                        time=time_slice,
                        latitude=slice(lat, lat + 1.0),
                        longitude=slice(lon_360, lon_360 + 1.0)
                    )
                    
                    if 'fishing_hours' not in ds_event.variables or 'thetao' not in ds_event.variables:
                        continue
                        
                    T_tensor = ds_event['thetao'].values
                    E_tensor = ds_event['fishing_hours'].values
                    
                    T_tensor = np.nan_to_num(T_tensor, nan=0.0)
                    E_tensor = np.nan_to_num(E_tensor, nan=0.0)
                    
                    q_tensor = calculer_q_matrice(T_tensor)
                    
                    taux_exploitation = q_tensor * E_tensor
                    taux_exploitation = np.clip(taux_exploitation, 0.0, 0.99)
                    
                    C_fine_tensor = taux_exploitation * B_p
                    
                    C_evenement = np.sum(C_fine_tensor)
                    C_annuel += C_evenement
                    
                    y_true_list.append(catch_reel)
                    y_pred_list.append(C_evenement)
                    
                    results_list.append({
                        'year': year,
                        'month': month,
                        'catch_reel': catch_reel,
                        'catch_predit': round(C_evenement, 2),
                        'lon': lon,
                        'lat': lat
                    })
                    
                except Exception:
                    continue
                    
        B_t = B_t + R_GROWTH * B_t * (1 - (B_t / K_CAPACITY)) - C_annuel
        
        if B_t < 0:
            B_t = 0.0
            
    df_results = pd.DataFrame(results_list)
    csv_save_path = os.path.join(save_dir, "sprfmo_predictions_vs_realite.csv")
    df_results.to_csv(csv_save_path, index=False)
    print(f"\nFichier CSV detaille enregistre: {csv_save_path}")
            
    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    print("\nResultats de l'evaluation")
    print(f"MAE  : {mae:,.0f} kg")
    print(f"RMSE : {rmse:,.0f} kg")
    print(f"R2   : {r2:.3f}")
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    max_val = max(y_true_arr.max(), y_pred_arr.max())
    
    sns.scatterplot(x=y_true_arr, y=y_pred_arr, ax=ax, alpha=0.4, color='b', edgecolor=None)
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Prediction parfaite")
    
    ax.set_title("Modele Halieutique Dynamique Spatialise", fontsize=14, fontweight='bold')
    ax.set_xlabel("Capture Reelle SPRFMO (kg)", fontsize=12)
    ax.set_ylabel("Capture Predite (kg)", fontsize=12)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    textstr = '\n'.join((
        f'MAE  = {mae:,.0f} kg',
        f'RMSE = {rmse:,.0f} kg',
        f'R2   = {r2:.3f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "comparaison_modele_dynamique.png")
    plt.savefig(save_path, dpi=300)
    print(f"Graphique enregistre: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_simulation_and_evaluate(PATH_CSV, DIR_TENSORS, SAVE_DIR)

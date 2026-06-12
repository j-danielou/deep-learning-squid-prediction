# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:46:18 2026

@author: jdanielou
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import scipy.special as sp

k=9
T = np.linspace(3,40,200)
# T_max = 30
# T_adj = T_max - T

t_shift = 12
T_adj = T - t_shift

T_safe = T_safe = np.where(T_adj > 0, T_adj, 0.0)
        
numerator = (T_safe ** (k/2.0 - 1.0)) * np.exp(-T_safe / 2.0)
denominator = (2.0 ** (k/2.0)) * sp.gamma(k/2.0)

        
chi2_pdf = numerator / denominator

q = 5* chi2_pdf

plt.plot(T,q)

#%%

import matplotlib.pyplot as plt
import numpy as np

T = np.linspace(0, 40, 200)

mu = 19.0
sigma = 4.0

q = np.exp(-0.5 * ((T - mu) / sigma) ** 2)

plt.plot(T, q)
plt.axvline(x=5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=30, color='r', linestyle='--', alpha=0.3)
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

# Définition des paramètres de votre fonction
mu = 18          # Optimum
sigma_montee = 5.50  # Largeur pour T <= mu (montée rapide)
sigma_descente = 5.0 # Largeur pour T > mu (descente lente)
amplitude = 0.6     # Valeur maximale à l'optimum

# Plage de température
T = np.linspace(0, 40, 200)

# Calcul de la fonction asymétrique
q = amplitude * np.piecewise(T,
    [T <= mu, T > mu],
    [lambda t: np.exp(-0.5 * ((t - mu) / sigma_montee)**2),
     lambda t: np.exp(-0.5 * ((t - mu) / sigma_descente)**2)]
)

# Visualisation
plt.plot(T, q)
plt.title(f"Fonction Asymétrique\nOptimum={mu}°, Montée={sigma_montee}, Descente={sigma_descente}")
plt.xlabel("Température (°C)")
plt.ylabel("q")
plt.grid(True, linestyle='--', alpha=0.5)
plt.axvline(x=mu, color='r', linestyle=':', label=f'Optimum ({mu}°)')
plt.legend()
plt.show()

#%%

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

file_path = r"D:/deep-learning-squid-prediction/data/tensors_annuels/phy/tensor_2012.nc"
test_ds = xr.open_dataset(file_path)

#%%

PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"

from dataloader import MILFisheryDataset
train_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_train, FEATURES, PATH_MASK, is_train=True)



E=10 # à modifier 
bt= 6e9
bt_l=[bt]
i=0
r=1.2
k=2e11
for t in range(12):
    bt= r*bt*(1-(bt/k)) - q*E*bt
    bt_l.append(bt)
    i=i+1
    print(i)


plt.plot(bt_l)


#%%

import numpy as np
import matplotlib.pyplot as plt
from dataloader import MILFisheryDataset

# Chemins et chargement des donnees
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"

# train_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_train, FEATURES, PATH_MASK, is_train=True)

def calculer_q(temperature_actuelle):
    # Parametres de la fonction asymetrique
    mu = 18.0
    sigma_montee = 5.50
    sigma_descente = 5.0
    amplitude = 0.6
    
    if temperature_actuelle <= mu:
        return amplitude * np.exp(-0.5 * ((temperature_actuelle - mu) / sigma_montee)**2)
    else:
        return amplitude * np.exp(-0.5 * ((temperature_actuelle - mu) / sigma_descente)**2)

# Initialisation du modele de biomasse
E = 10
bt = 6e9
r = 1.2
k = 2e11
annees = 12

bt_l = [bt]
q_l = []

# Temperatures annuelles (a remplacer par les donnees extraites de train_ds)
temperatures_annuelles = [15.5, 16.2, 17.8, 18.1, 19.5, 18.0, 17.2, 16.5, 15.8, 17.0, 18.5, 19.2]

# Boucle de simulation
print("Annee | Temp | q | Biomasse")
print("---------------------------")

for t in range(annees):
    T = temperatures_annuelles[t]
    
    # Mise a jour de q en fonction de la temperature de l'annee
    q = calculer_q(T)
    q_l.append(q)
    
    # Mise a jour de la biomasse
    bt = r * bt * (1 - (bt / k)) - q * E * bt
    
    # Securite pour eviter les valeurs negatives
    if bt < 0:
        bt = 0
        
    bt_l.append(bt)
    print(f"{t+1:5} | {T:4.1f} | {q:.4f} | {bt:.2e}")

# Visualisation des resultats
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(bt_l, marker='o', color='blue')
ax1.set_title("Evolution de la biomasse (bt)")
ax1.set_ylabel("Biomasse")
ax1.grid(True)

ax2.plot(range(1, annees + 1), q_l, marker='x', color='red')
ax2.set_title("Evolution de la capturabilite (q)")
ax2.set_xlabel("Annee")
ax2.set_ylabel("Valeur de q")
ax2.grid(True)

plt.tight_layout()
plt.show()

#%%
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

# Parametres de capturabilite
MU = 18.0
SIGMA_MONTEE = 5.50
SIGMA_DESCENTE = 5.0
AMPLITUDE = 0.001 # Ajuste pour correspondre a l'efficacite par heure de peche

# Parametres halieutiques (Ordre de grandeur SPRFMO pour Dosidicus gigas)
R_GROWTH = 1.2
K_CAPACITY = 10e9 # 10 millions de tonnes
B_INITIAL = 5e9   # 5 millions de tonnes

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
    
    print("Demarrage de la simulation dynamique...")
    
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
                    
                    # Verrouillage physique du taux d'exploitation (max 99%)
                    taux_exploitation = q_tensor * E_tensor
                    taux_exploitation = np.clip(taux_exploitation, 0.0, 0.99)
                    
                    C_fine_tensor = taux_exploitation * B_p
                    
                    C_evenement = np.sum(C_fine_tensor)
                    C_annuel += C_evenement
                    
                    y_true_list.append(catch_reel)
                    y_pred_list.append(C_evenement)
                    
                except Exception:
                    continue
                    
        # Mise a jour de la biomasse
        B_t = B_t + R_GROWTH * B_t * (1 - (B_t / K_CAPACITY)) - C_annuel
        
        if B_t < 0:
            B_t = 0.0
            
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
    print(f"\nGraphique enregistre: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_simulation_and_evaluate(PATH_CSV, DIR_TENSORS, SAVE_DIR)

#%%
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
DIR_TENSORS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
SAVE_DIR = "D:/deep-learning-squid-prediction/data/"

MU = 18.0
SIGMA_MONTEE = 5.50
SIGMA_DESCENTE = 5.0

def calculer_q_matrice(T_matrice, amplitude):
    mask_inf = T_matrice <= MU
    mask_sup = T_matrice > MU
    
    q = np.zeros_like(T_matrice)
    q[mask_inf] = amplitude * np.exp(-0.5 * ((T_matrice[mask_inf] - MU) / SIGMA_MONTEE)**2)
    q[mask_sup] = amplitude * np.exp(-0.5 * ((T_matrice[mask_sup] - MU) / SIGMA_DESCENTE)**2)
    
    return q

def preparer_cache_donnees(csv_path, tensors_dir):
    """Extrait et stocke les tenseurs en memoire pour accelerer l'optimisation"""
    print("Chargement des donnees SPRFMO...")
    df = pd.read_csv(csv_path)
    df_gis = df[df['species_code'] == 'GIS'].copy()
    
    years = sorted(df_gis['year'].unique())
    cache_simulation = {}
    
    print("Mise en cache des tenseurs physiques...")
    for year in years:
        tensor_path = os.path.join(tensors_dir, f"tensor_{year}.nc")
        if not os.path.exists(tensor_path):
            continue
            
        df_year = df_gis[df_gis['year'] == year]
        N_events = len(df_year)
        if N_events == 0:
            continue
            
        evenements_annee = []
        
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
                        
                    T_tensor = np.nan_to_num(ds_event['thetao'].values, nan=0.0)
                    E_tensor = np.nan_to_num(ds_event['fishing_hours'].values, nan=0.0)
                    
                    evenements_annee.append({
                        'catch_reel': catch_reel,
                        'T': T_tensor,
                        'E': E_tensor
                    })
                except Exception:
                    continue
                    
        if evenements_annee:
            cache_simulation[year] = evenements_annee
            
    return cache_simulation

def simuler_modele(params, cache_donnees):
    """Execute la simulation biologique sur les donnees en cache"""
    amplitude, r, K, B_init = params
    
    y_true = []
    y_pred = []
    B_t = B_init
    
    for year in sorted(cache_donnees.keys()):
        evenements = cache_donnees[year]
        N_events = len(evenements)
        
        B_p = B_t / (N_events * 144)
        C_annuel = 0.0
        
        for ev in evenements:
            q_tensor = calculer_q_matrice(ev['T'], amplitude)
            taux_exploitation = np.clip(q_tensor * ev['E'], 0.0, 0.99)
            
            C_evenement = np.sum(taux_exploitation * B_p)
            C_annuel += C_evenement
            
            y_true.append(ev['catch_reel'])
            y_pred.append(C_evenement)
            
        B_t = B_t + r * B_t * (1 - (B_t / K)) - C_annuel
        if B_t < 0:
            B_t = 0.0
            
    return np.array(y_true), np.array(y_pred)

def fonction_objectif(params, cache_donnees):
    """Fonction que l'optimiseur cherche a minimiser (MSE)"""
    y_true, y_pred = simuler_modele(params, cache_donnees)
    if len(y_true) == 0:
        return 1e18
    return mean_squared_error(y_true, y_pred)

if __name__ == "__main__":
    # 1. Preparation des donnees
    cache_donnees = preparer_cache_donnees(PATH_CSV, DIR_TENSORS)
    
    # 2. Definition des parametres initiaux [AMPLITUDE, R_GROWTH, K_CAPACITY, B_INITIAL]
    p_init = [0.01, 1.2, 10e9, 5e9]
    
    # Bornes strictes pour chaque parametre (min, max)
    bornes = [
        (1e-6, 1.0),    # AMPLITUDE
        (0.1, 3.0),     # R_GROWTH
        (1e9, 50e9),    # K_CAPACITY
        (1e9, 20e9)     # B_INITIAL
    ]
    
    print("\nLancement de l'optimisation des parametres...")
    res = minimize(
        fonction_objectif, 
        p_init, 
        args=(cache_donnees,), 
        method='L-BFGS-B', 
        bounds=bornes
    )
    
    # 3. Extraction des meilleurs parametres
    meilleurs_params = res.x
    print("\nParametres optimises :")
    print(f"  AMPLITUDE  : {meilleurs_params[0]:.6f}")
    print(f"  R_GROWTH   : {meilleurs_params[1]:.3f}")
    print(f"  K_CAPACITY : {meilleurs_params[2]:.2e}")
    print(f"  B_INITIAL  : {meilleurs_params[3]:.2e}")
    
    # 4. Evaluation finale avec les parametres optimises
    y_true, y_pred = simuler_modele(meilleurs_params, cache_donnees)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\nResultats apres optimisation")
    print(f"  MAE  : {mae:,.0f} kg")
    print(f"  RMSE : {rmse:,.0f} kg")
    print(f"  R2   : {r2:.3f}")
    
    # 5. Graphique des resultats
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    max_val = max(y_true.max(), y_pred.max())
    
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.4, color='b', edgecolor=None)
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Prediction parfaite")
    
    ax.set_title("Modele Halieutique Dynamique Optimise", fontsize=14, fontweight='bold')
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
    save_path = os.path.join(SAVE_DIR, "comparaison_modele_optimise.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nGraphique enregistre: {save_path}")
    plt.show()

#%%
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
DIR_TENSORS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
SAVE_DIR = "D:/deep-learning-squid-prediction/data/"

MU = 18.0
SIGMA_MONTEE = 5.50
SIGMA_DESCENTE = 5.0

def calculer_q_matrice(T_matrice, amplitude):
    mask_inf = T_matrice <= MU
    mask_sup = T_matrice > MU
    
    q = np.zeros_like(T_matrice)
    q[mask_inf] = amplitude * np.exp(-0.5 * ((T_matrice[mask_inf] - MU) / SIGMA_MONTEE)**2)
    q[mask_sup] = amplitude * np.exp(-0.5 * ((T_matrice[mask_sup] - MU) / SIGMA_DESCENTE)**2)
    
    return q

def preparer_cache_donnees(csv_path, tensors_dir):
    print("Chargement des donnees SPRFMO...")
    df = pd.read_csv(csv_path)
    df_gis = df[df['species_code'] == 'GIS'].copy()
    
    years = sorted(df_gis['year'].unique())
    cache_simulation = {}
    
    print("Mise en cache des tenseurs physiques...")
    for year in years:
        tensor_path = os.path.join(tensors_dir, f"tensor_{year}.nc")
        if not os.path.exists(tensor_path):
            continue
            
        df_year = df_gis[df_gis['year'] == year]
        N_events = len(df_year)
        if N_events == 0:
            continue
            
        evenements_annee = []
        
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
                        
                    T_tensor = np.nan_to_num(ds_event['thetao'].values, nan=0.0)
                    E_tensor = np.nan_to_num(ds_event['fishing_hours'].values, nan=0.0)
                    
                    evenements_annee.append({
                        'catch_reel': catch_reel,
                        'T': T_tensor,
                        'E': E_tensor
                    })
                except Exception:
                    continue
                    
        if evenements_annee:
            cache_simulation[year] = evenements_annee
            
    return cache_simulation

def simuler_modele(params, cache_donnees):
    amplitude, r, K, B_init = params
    
    y_true = []
    y_pred = []
    B_t = B_init
    
    for year in sorted(cache_donnees.keys()):
        evenements = cache_donnees[year]
        N_events = len(evenements)
        
        # Securite pour eviter les divisions par zero ou valeurs aberrantes
        if N_events == 0 or B_t <= 0:
            B_t = B_t + r * B_t * (1 - (B_t / K)) if B_t > 0 else 0
            continue
            
        B_p = B_t / (N_events * 144)
        C_annuel = 0.0
        
        for ev in evenements:
            q_tensor = calculer_q_matrice(ev['T'], amplitude)
            taux_exploitation = np.clip(q_tensor * ev['E'], 0.0, 0.99)
            
            C_evenement = np.sum(taux_exploitation * B_p)
            C_annuel += C_evenement
            
            y_true.append(ev['catch_reel'])
            y_pred.append(C_evenement)
            
        B_t = B_t + r * B_t * (1 - (B_t / K)) - C_annuel
        if B_t < 0:
            B_t = 0.0
            
    return np.array(y_true), np.array(y_pred)

def fonction_objectif(params, cache_donnees):
    y_true, y_pred = simuler_modele(params, cache_donnees)
    if len(y_true) == 0:
        return 1e12
    # Utilisation du RMSE au lieu du MSE pour eviter l'explosion numerique
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    cache_donnees = preparer_cache_donnees(PATH_CSV, DIR_TENSORS)
    
    # Bornes strictes : [AMPLITUDE, R_GROWTH, K_CAPACITY, B_INITIAL]
    bornes = [
        (1e-5, 0.1),    # AMPLITUDE (On donne plus d'espace de recherche vers le haut)
        (0.5, 3.0),     # R_GROWTH
        (5e9, 50e9),    # K_CAPACITY
        (1e9, 20e9)     # B_INITIAL
    ]
    
    print("\nLancement de l'optimisation par Evolution Differentielle (Patientez quelques minutes)...")
    
    # L'evolution differentielle est insensible aux problemes d'echelle
    res = differential_evolution(
        fonction_objectif, 
        bounds=bornes,
        args=(cache_donnees,),
        strategy='best1bin',
        popsize=15,       # Taille de la population (15 * 4 variables = 60 modeles par generation)
        tol=0.01,         # Tolerance d'arret
        mutation=(0.5, 1.5),
        recombination=0.7,
        disp=True         # Affiche la progression generation par generation dans la console
    )
    
    meilleurs_params = res.x
    print("\n==============================")
    print("      PARAMETRES OPTIMISES    ")
    print("==============================")
    print(f"  AMPLITUDE  : {meilleurs_params[0]:.6f}")
    print(f"  R_GROWTH   : {meilleurs_params[1]:.3f}")
    print(f"  K_CAPACITY : {meilleurs_params[2]:.2e}")
    print(f"  B_INITIAL  : {meilleurs_params[3]:.2e}")
    
    y_true, y_pred = simuler_modele(meilleurs_params, cache_donnees)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n==============================")
    print("    METRIQUES FINALES (TEST)  ")
    print("==============================")
    print(f"  MAE  : {mae:,.0f} kg")
    print(f"  RMSE : {rmse:,.0f} kg")
    print(f"  R2   : {r2:.3f}")
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    max_val = max(y_true.max(), y_pred.max())
    
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.4, color='b', edgecolor=None)
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Prediction parfaite")
    
    ax.set_title("Modele Halieutique - Apres Evolution Differentielle", fontsize=14, fontweight='bold')
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
    save_path = os.path.join(SAVE_DIR, "comparaison_modele_optimise_diff.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nGraphique enregistre: {save_path}")
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:25:46 2026

@author: jdanielou
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset

from model import WeeklyMILModel

#config
ARCHITECTURE_NAME = "SwinLSTM_TransMIL_asy" 
MODEL_PATH = "D:/deep-learning-squid-prediction/models/model_swinlstm_transmil_hybride_20260601_165620_mae398659.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil de calcul : {device}")
print(f"Analyse d'explicabilite globale pour : {ARCHITECTURE_NAME}")

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
DOSSIER_PLOTS = "D:/deep-learning-squid-prediction/models/explainability/"
os.makedirs(DOSSIER_PLOTS, exist_ok=True)

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

print("\nChargement des donnees de validation (Annee 2024)...")
df_all = load_and_prepare_catch_data(PATH_CSV)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)

val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

print("Chargement du modele")
model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=512).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

#function
def evaluate_mae(model, loader, feature_to_ablate=None):
    """
    Calcule le MAE sur toute l'annee. 
    Si feature_to_ablate est fourni, met cette variable a zero partout.
    """
    total_error_kg = 0.0
    total_weight = 0.0
    
    with torch.no_grad():
        for x, y, w, m, t_mask in tqdm(loader, desc=f"Eval {feature_to_ablate or 'Baseline'}", leave=False):
            x, y, w, m, t_mask = x.to(device), y.to(device), w.to(device), m.to(device), t_mask.to(device)
            
            #ablation
            if feature_to_ablate is not None:
                feat_idx = FEATURES.index(feature_to_ablate)
                x[:, feat_idx, :, :, :] = 0.0 
                
            y_pred, _ = model(x, pixel_mask=m, time_mask=t_mask)
            
            safe_pred = torch.clamp(y_pred.squeeze(1), min=0.0, max=20.0)
            pred_kg = torch.expm1(safe_pred)
            true_kg = torch.expm1(y)
            
            abs_error = torch.abs(pred_kg - true_kg)
            total_error_kg += (abs_error * w).sum().item()
            total_weight += w.sum().item()
            
    return total_error_kg / max(1e-8, total_weight)

print("\n[1/3] Calcul du MAE de reference (Baseline)")
baseline_mae = evaluate_mae(model, val_loader)
print(f"MAE de reference : {baseline_mae:,.0f} kg")

print("\n[2/3] Calcul de l'importance globale des variables (Ablation)...")
results = []
for feat in FEATURES:
    ablated_mae = evaluate_mae(model, val_loader, feature_to_ablate=feat)
    impact_kg = ablated_mae - baseline_mae
    results.append({'Variable': feat, 'Impact_MAE_kg': impact_kg})
    print(f"  -> Sans {feat.ljust(15)} : Impact = {impact_kg:+,.0f} kg")

print("\n[3/3] Generation des rapports et graphiques")
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='Impact_MAE_kg', ascending=False)

#Save CSV
csv_path = os.path.join(DOSSIER_PLOTS, f"importance_variables_{ARCHITECTURE_NAME}.csv")
df_results.to_csv(csv_path, index=False)

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

colors = ['#e74c3c' if val > 0 else '#3498db' for val in df_results['Impact_MAE_kg']]

ax = sns.barplot(x='Impact_MAE_kg', y='Variable', data=df_results, palette=colors)

plt.title(f"Importance Globale des Variables (Annee 2024)\nArchitecture : {ARCHITECTURE_NAME}", fontsize=14, pad=20)
plt.xlabel("Impact sur l'erreur (kg)\n(Valeurs Positives = Le modèle se dégrade = Variable Utile)", fontsize=12)
plt.ylabel("Variables Océanographiques / Humaines", fontsize=12)

for i, v in enumerate(df_results['Impact_MAE_kg']):
    ax.text(v, i, f" {v:+,.0f} kg", va='center', fontsize=10, 
            color='black' if v > 0 else 'blue', fontweight='bold')

plt.axvline(0, color='black', linewidth=1.5)
plt.tight_layout()

plot_path = os.path.join(DOSSIER_PLOTS, f"importance_variables_{ARCHITECTURE_NAME}.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nMission terminee ! Le rapport est sauvegarde dans : {DOSSIER_PLOTS}")
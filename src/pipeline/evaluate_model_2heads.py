# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:09:47 2026

@author: jdanielou
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset

dossier_actuel = os.path.dirname(os.path.abspath(__file__))
dossier_parent = os.path.dirname(dossier_actuel)
dossier_marbec = os.path.join(dossier_parent, "MARBEC-GPU")
if dossier_marbec not in sys.path:
    sys.path.insert(0, dossier_marbec)
    
from model_2heads_cluster import WeeklyMILModel

#conf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil de calcul : {device}")

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"

#variables
FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

#modèles à comparer
MODELS_TO_EVALUATE = {
    #"Baseline(Conv3d + Transformers)": "D:/deep-learning-squid-prediction/models/model_baseline_ancien.pth",
    #"Hybride TransMIL(Conv3d + TransMIL)": "D:/deep-learning-squid-prediction/models/model_conv3d_transmil_hybride_20260522_203142_mae332721.pth"
    "SwinLSTM TransMIL(SwinLSTM + TransMIL)": "D:/deep-learning-squid-prediction/models/meilleur_modele_optuna_ziln_V4.pth"
}

#dataloading
print("Chargement des donnees de validation")
df_all = load_and_prepare_catch_data(PATH_CSV)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)

print(f"Nombre d'evenements pour l'evaluation : {len(df_val)}")

val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

#Evaluation
def evaluate_model(model_path, model_name):
    print(f"\n--- Evaluation de : {model_name} ---")
    
    #initialisation
    model = WeeklyMILModel(
        in_channels=len(FEATURES), 
        hidden_dim=512,
        num_conv_layers=3,
        num_fc_layers=2
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"ATTENTION : Le fichier {model_path} est introuvable. Modele ignore.")
        return None
        
    model.eval()
    
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for x, y, w, m, t_mask in val_loader:
            x = x.to(device)
            m = m.to(device)
            t_mask = t_mask.to(device)
            
            # ADAPTATION ZILN : Récupération des deux têtes
            logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
            
            # ADAPTATION ZILN : Reconversion mathématique Probabilité * Quantité
            prob_presence = torch.sigmoid(logit_presence.cpu()).squeeze()
            safe_pred = torch.clamp(pred_abundance.cpu(), min=0.0, max=20.0).squeeze()
            
            pred_kg = (prob_presence * torch.expm1(safe_pred)).numpy()
            true_kg = torch.expm1(y).numpy()
            weights = w.numpy()
            
            # Gestion des scalaires si le dernier batch ne contient qu'un seul élément
            if pred_kg.ndim == 0:
                pred_kg = np.array([pred_kg])
                true_kg = np.array([true_kg])
                weights = np.array([weights])
            
            # On ne garde que les blocs valides (hors ZEE pure, où w > 0)
            for p, t, weight in zip(pred_kg, true_kg, weights):
                if weight > 0:
                    y_pred_list.append(p)
                    y_true_list.append(t)
                    
    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)
    
    # Calcul des metriques
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    print(f"MAE  : {mae:,.0f} kg")
    print(f"RMSE : {rmse:,.0f} kg")
    print(f"R2   : {r2:.3f}")
    
    return y_true_arr, y_pred_arr, mae, rmse, r2

# --- 4. EXECUTION ET CARTOGRAPHIE DE COMPARAISON ---
results = {}
for name, path in MODELS_TO_EVALUATE.items():
    res = evaluate_model(path, name)
    if res is not None:
        results[name] = res

# Génération de la figure de comparaison
if len(results) > 0:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6), squeeze=False)
    
    # Ligne idéale y=x (pour calculer l'échelle max du graphique)
    max_val = max([max(res[0].max(), res[1].max()) for res in results.values()])
    
    for idx, (name, (y_true, y_pred, mae, rmse, r2)) in enumerate(results.items()):
        ax = axes[0, idx]
        
        # Nuage de points
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.4, color='b', edgecolor=None)
        
        # Ligne de perfection (y=x)
        ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label="Prediction parfaite")
        
        # Formatage des axes (en milliers de tonnes pour la lisibilité)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Capture Réelle (kg)", fontsize=12)
        ax.set_ylabel("Capture Prédite (kg)", fontsize=12)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        
        # Ajout du bloc de texte avec les métriques
        textstr = '\n'.join((
            f'MAE  = {mae:,.0f} kg',
            f'RMSE = {rmse:,.0f} kg',
            f'R²   = {r2:.3f}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    plt.savefig(os.path.join("D:/deep-learning-squid-prediction/data", "comparaison_modeles_SwinLSTM_TransMIL_2heads_optuna.png"), dpi=300)
    print("\nGraphique de comparaison sauvegarde")
    plt.show()
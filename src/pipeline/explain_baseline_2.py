# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:45:11 2026

@author: jdanielou
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import os

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model import WeeklyMILModel

# Configuration
device = torch.device("cpu")
PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
MODEL_PATH = "D:/deep-learning-squid-prediction/models/model_conv3d_transmil_hybride_20260522_203142_mae332721.pth"

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

print("Chargement des donnees de validation (2024)")
df_all = load_and_prepare_catch_data(PATH_CSV)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)
val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)

#Batch size de 1 pour isoler un seul evenement
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True) 

#Chargement du modele TransMIL (hidden_dim=256)
print("Chargement du modele TransMIL")
model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=512).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

#Extraction d'un bloc oceanique
x, y, w, m, t_mask = next(iter(val_loader))

with torch.no_grad():
    #Inférence
    y_pred, att_weights = model(x, pixel_mask=m, time_mask=t_mask)
    
    #Conversion log vers kilos reels
    pred_kg = torch.expm1(torch.clamp(y_pred, min=0.0, max=20.0)).item()
    true_kg = torch.expm1(y).item()
    
    #Reshape des poids (576 instances -> 4 semaines de 12x12 pixels)
    attention_map = att_weights.view(4, 12, 12).numpy()
    
    #Heat maps
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Attention TransMIL (Vraie Capture: {true_kg:,.0f} kg | Prediction: {pred_kg:,.0f} kg)", fontsize=14)
    
    for i in range(4):
        sns.heatmap(attention_map[i], ax=axes[i], cmap="jet", vmin=0, vmax=attention_map.max(), 
                    cbar=(i==3), square=True)
        axes[i].set_title(f"Semaine {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    #IMPORTANCE DES VARIABLES
    print("\nCalcul de l'importance des variables (Ablation)")
    base_error = abs(pred_kg - true_kg)
    feature_importance = {}

    for i, feat in enumerate(FEATURES):
        x_ablated = x.clone()
        #Masquage de la variable
        x_ablated[:, i, :, :, :] = 0.0 
        
        y_pred_ablated, _ = model(x_ablated, pixel_mask=m, time_mask=t_mask)
        pred_kg_ablated = torch.expm1(torch.clamp(y_pred_ablated, min=0.0, max=20.0)).item()
        error_ablated = abs(pred_kg_ablated - true_kg)
        
        importance = error_ablated - base_error
        feature_importance[feat] = importance

    #Affichage classement
    sorted_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    print("\nClassement des variables (Impact sur l'erreur en kilos) :")
    for feat, imp in sorted_features:
        print(f"{feat.ljust(15)} : {imp:+,.0f} kg d'erreur si supprimee")
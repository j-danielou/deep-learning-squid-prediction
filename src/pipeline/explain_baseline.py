# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:21:31 2026

@author: jdanielou
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model import WeeklyMILModel

#conf
device = torch.device("cpu")
PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
MODEL_PATH = "D:/deep-learning-squid-prediction/models/best_squid_nowcasting_model.pth"

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

print("Chargement des donnees de validation (2024)")
df_all = load_and_prepare_catch_data(PATH_CSV)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)
val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)


val_loader = DataLoader(val_ds, batch_size=1, shuffle=True) 

#model
print("Chargement du meilleur modele...")
model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

#event
x, y, w, m, t_mask = next(iter(val_loader))

with torch.no_grad():
    #heatmap attention
    y_pred, _ = model(x, pixel_mask=m, time_mask=t_mask)
 
    features = F.relu(model.encoder(x))
    weekly_features = model.temporal_pool(features, t_mask)
    b, c, t, h, w_dim = weekly_features.size()
    mil_instances = weekly_features.view(b, c, -1).transpose(1, 2)
    
    att_weights = model.attention(mil_instances)
    
    #softmax pour les proba
    mask_expanded = m.unsqueeze(1).expand(-1, t, -1, -1).reshape(b, 576, 1)
    att_weights = att_weights.masked_fill(mask_expanded == 0, -1e9)
    att_weights = torch.softmax(att_weights, dim=1)
    
    # Reshape en 4 semaines de 12x12 pixels
    attention_map = att_weights.view(4, 12, 12).numpy()
    
    #Heat maps
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Attention du réseau (Vraie Capture: {torch.expm1(y).item():.0f} kg | Prédiction: {torch.expm1(y_pred).item():.0f} kg)", fontsize=14)
    
    for i in range(4):
        sns.heatmap(attention_map[i], ax=axes[i], cmap="jet", vmin=0, vmax=attention_map.max(), 
                    cbar=(i==3), square=True)
        axes[i].set_title(f"Semaine {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    #features ablation
    print("\nCalcul de l'importance des variables (Ablation)...")
    base_error = torch.abs(torch.expm1(y_pred) - torch.expm1(y)).item()
    feature_importance = {}

    for i, feat in enumerate(FEATURES):
        x_ablated = x.clone()
        # On met toute la variable à zéro (la moyenne du Z-score)
        x_ablated[:, i, :, :, :] = 0.0 
        
        y_pred_ablated, _ = model(x_ablated, pixel_mask=m, time_mask=t_mask)
        error_ablated = torch.abs(torch.expm1(y_pred_ablated) - torch.expm1(y)).item()
       
        importance = error_ablated - base_error
        feature_importance[feat] = importance

    sorted_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    print("\nClassement des variables (Impact sur la MAE en kilos) :")
    for feat, imp in sorted_features:
        print(f"{feat.ljust(15)} : + {imp:,.0f} kg d'erreur si supprimee")
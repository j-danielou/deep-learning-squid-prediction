# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:50:45 2026

@author: jdanielou
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model import WeeklyMILModel

#Config
device = torch.device("cpu")
PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"

MODEL_PATH = "D:/deep-learning-squid-prediction/models/model_swinlstm_transmil_hybride_20260601_165620_mae398659.pth" 

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

MOIS_CIBLE = 1
ANNEE_CIBLE = 2024

print("Chargement des donnees")
df_all = load_and_prepare_catch_data(PATH_CSV)
df_val = df_all[(df_all['year'] == ANNEE_CIBLE) & (df_all['month'] == MOIS_CIBLE)].reset_index(drop=True)

if len(df_val) == 0:
    raise ValueError("Aucune donnee de peche pour ce mois.")

print(f"Blocs exploites : {len(df_val)}")

val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False) 

print("Chargement du modele")
model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=512).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

LAT_MIN, LAT_MAX = -38.0, 3.0
LON_MIN, LON_MAX = 227.0, 293.0

LAT_PIXELS = int(LAT_MAX - LAT_MIN) * 12 
LON_PIXELS = int(LON_MAX - LON_MIN) * 12 

#Init
global_attention_map = np.full((4, LAT_PIXELS, LON_PIXELS), np.nan)

print("Inference spatiale en cours...")
with torch.no_grad():
    for i, (x, y, w, m, t_mask) in enumerate(val_loader):
        y_pred, att_weights = model(x, pixel_mask=m, time_mask=t_mask)
        
        row = df_val.iloc[i]
        lat = row.lat
        lon = row.lon_360
        
        lat_idx = int(lat - LAT_MIN) * 12
        lon_idx = int(lon - LON_MIN) * 12
        
        #Reshape 4 semaines
        attention_patch = att_weights.view(4, 12, 12).cpu().detach().numpy()
        global_attention_map[:, lat_idx:lat_idx+12, lon_idx:lon_idx+12] = attention_patch

print("Generation des graphiques...")
fig, axes = plt.subplots(1, 4, figsize=(24, 8), facecolor='white', 
                         subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"Dynamique d'Attention TransMIL - ({MOIS_CIBLE:02d}/{ANNEE_CIBLE})", fontsize=20, fontweight='bold')

extent_cartopy = [LON_MIN - 360, LON_MAX - 360, LAT_MIN, LAT_MAX]

for week in range(4):
    ax = axes[week]
    ax.set_extent(extent_cartopy, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=0) 
    ax.add_feature(cfeature.LAND, facecolor='darkgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black', zorder=3)
    
    im = ax.imshow(global_attention_map[week], cmap='jet', origin='lower', extent=extent_cartopy, 
                   transform=ccrs.PlateCarree(), vmin=0, vmax=0.05, zorder=1)
    
    ax.set_title(f"Semaine {week+1}", fontsize=16)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='black', alpha=0.3, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.5, pad=0.1)
cbar.set_label("Poids d'Attention", fontsize=14)

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:13:52 2026

@author: jdanielou
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import optuna.visualization as vis
from torch.utils.data import DataLoader
import numpy as np

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model import WeeklyMILModel

#config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil de calcul : {device}")

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"
DOSSIER_MODELS = "D:/deep-learning-squid-prediction/models/"

os.makedirs(DOSSIER_MODELS, exist_ok=True)

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

#dataloading
print("Chargement en memoire des donnees")
df_all = load_and_prepare_catch_data(PATH_CSV)

#RandomSampliing
df_train_complet = df_all[(df_all['year'] >= 2012) & (df_all['year'] <= 2023)]
df_train = df_train_complet.sample(frac=0.2, random_state=42).reset_index(drop=True)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)

print(f"Taille de l'echantillon d'entrainement reduit pour Optuna : {len(df_train)} blocs")

def objective(trial):
    """
    Fonction objectif pour Optuna.
    Teste une combinaison spécifique d'hyperparamètres.
    """
    #espace de recherche
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    print(f"\n>>> Lancement Essai #{trial.number}")
    print(f"Params : Opt={optimizer_name}, Dim={hidden_dim}, Batch={batch_size}, LR={lr:.5f}")

    #Initialisation
    train_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_train, FEATURES, PATH_MASK, is_train=True)
    val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=hidden_dim).to(device)
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.L1Loss(reduction='none')

    #training
    epochs = 10
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for x, y, w, m, t_mask in train_loader:
            x, y, w, m, t_mask = x.to(device), y.to(device), w.to(device), m.to(device), t_mask.to(device)
            
            optimizer.zero_grad()
            y_pred, _ = model(x, pixel_mask=m, time_mask=t_mask)
            
            loss_unreduced = criterion(y_pred.squeeze(), y)
            loss = (loss_unreduced * w).mean()
            
            loss.backward()
            optimizer.step()

        #val
        model.eval()
        val_loss_sum = 0
        total_weight = 0
        
        with torch.no_grad():
            for x, y, w, m, t_mask in val_loader:
                x, y, w, m, t_mask = x.to(device), y.to(device), w.to(device), m.to(device), t_mask.to(device)
                
                y_pred, _ = model(x, pixel_mask=m, time_mask=t_mask)
                
                pred_kg = torch.expm1(torch.clamp(y_pred, min=0.0, max=20.0)).squeeze()
                true_kg = torch.expm1(y)
                
                abs_diff = torch.abs(pred_kg - true_kg)
                val_loss_sum += (abs_diff * w).sum().item()
                total_weight += w.sum().item()
                
        val_loss = val_loss_sum / (total_weight + 1e-8)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"  - Epoque {epoch+1:02d}/{epochs} | Val MAE : {val_loss:,.0f} kg")

        #Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print("  -> Essai coupe prématurément (Pruning) pour mauvaises performances.")
            raise optuna.TrialPruned()

    return best_val_loss


if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEMARRAGE DE L'OPTIMISATION OPTUNA")
    print("="*50)
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    study.optimize(objective, n_trials=20)

    print("\n" + "="*50)
    print("RESULTATS DE L'OPTIMISATION")
    print("="*50)
    print(f"Meilleure erreur (MAE) atteinte : {study.best_value:,.0f} kg")
    print("\nMeilleurs hyperparametres trouves :")
    for key, value in study.best_trial.params.items():
        print(f"  - {key}: {value}")

    #Graphs
    print("\nGeneration des graphiques")
    
    fig_history = vis.plot_optimization_history(study)
    fig_history.write_html(os.path.join(DOSSIER_MODELS, "optuna_history.html"))

    fig_param_importances = vis.plot_param_importances(study)
    fig_param_importances.write_html(os.path.join(DOSSIER_MODELS, "optuna_importances.html"))

    fig_parallel = vis.plot_parallel_coordinate(study)
    fig_parallel.write_html(os.path.join(DOSSIER_MODELS, "optuna_parallel.html"))

    print(f"\nTermine ! Ouvre les fichiers .html dans ton dossier :\n{DOSSIER_MODELS}")
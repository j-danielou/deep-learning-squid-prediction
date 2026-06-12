# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:58:33 2026

@author: jdanielou
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import mlflow

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model_2heads import WeeklyMILModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil de calcul detecte : {device}")

BASE_DIR = "/marbec-data/Osmose-Montpellier/Lou-Jules"
PATH_CSV = os.path.join(BASE_DIR, "data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv")
PATH_PHYS = os.path.join(BASE_DIR, "data/tensors_annuels/phy/")
PATH_CHL = os.path.join(BASE_DIR, "data/tensors_annuels/bio/")
PATH_MASK = os.path.join(BASE_DIR, "data/official_mask_zee.nc")
DOSSIER_MODELS = os.path.join(BASE_DIR, "models/")
MLFLOW_DIR = os.path.join(BASE_DIR, "runs/mlflow")

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

MAX_EPOCHS = 25  
BATCH_SIZE = 16
N_TRIALS = 50  

DB_NAME = f"sqlite:///{os.path.join(DOSSIER_MODELS, 'optuna_ziln_study.db')}"
BEST_MODEL_PATH = os.path.join(DOSSIER_MODELS, "meilleur_modele_optuna_ziln.pth")
CSV_RESULTS_PATH = os.path.join(DOSSIER_MODELS, "resultats_optuna_ziln.csv")

os.makedirs(DOSSIER_MODELS, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Configuration de la base de stockage MLflow
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("squid_prediction_ziln")

print("Chargement des donnees")
df_all = load_and_prepare_catch_data(PATH_CSV)

df_train = df_all[(df_all['year'] >= 2012) & (df_all['year'] <= 2023)].reset_index(drop=True)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)

print(f"Taille entrainement : {len(df_train)} evenements")
print(f"Taille validation : {len(df_val)} evenements")

train_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_train, FEATURES, PATH_MASK, is_train=True)
val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

class ZeroInflatedLoss(nn.Module):
    def __init__(self, alpha_reg=1.0, delta_huber=1.0):
        super(ZeroInflatedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.huber = nn.HuberLoss(reduction='none', delta=delta_huber)
        self.alpha = alpha_reg

    def forward(self, logit_presence, pred_abundance, y_true, w):
        is_present = (y_true > 0).float()
        loss_cls = self.bce(logit_presence, is_present)
        
        loss_reg = self.huber(pred_abundance, y_true)
        loss_reg = loss_reg * is_present
        
        sum_w = w.sum() + 1e-8
        final_loss_cls = (loss_cls * w).sum() / sum_w
        final_loss_reg = (loss_reg * w).sum() / sum_w
        
        return final_loss_cls, final_loss_reg * self.alpha

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    alpha_reg = trial.suggest_float("alpha_reg", 0.5, 5.0)
    delta_huber = trial.suggest_float("delta_huber", 0.5, 3.0)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])

    model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ZeroInflatedLoss(alpha_reg=alpha_reg, delta_huber=delta_huber)

    best_local_mae = float('inf')

    try:
        global_best_mae = trial.study.best_value
    except ValueError:
        global_best_mae = float('inf')

    # Initialisation du suivi pour l'essai courant
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)

        for epoch in range(MAX_EPOCHS):
            model.train()
            total_loss, total_cls, total_reg, steps = 0.0, 0.0, 0.0, 0
            
            for x, y, w, m, t_mask in train_loader:
                x, y, w, m, t_mask = x.to(device), y.to(device), w.to(device), m.to(device), t_mask.to(device)
                optimizer.zero_grad()
                
                logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
                loss_cls, loss_reg = criterion(logit_presence.squeeze(1), pred_abundance.squeeze(1), y, w)
                loss = loss_cls + loss_reg
                
                if w.sum() > 0:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_cls += loss_cls.item()
                    total_reg += loss_reg.item()
                    steps += 1

            train_loss = total_loss / max(1, steps)
            train_cls = total_cls / max(1, steps)
            train_reg = total_reg / max(1, steps)

            model.eval()
            running_mae = 0.0
            count = 0
            with torch.no_grad():
                for x, y, w, m, t_mask in val_loader:
                    x, y, w, m, t_mask = x.to(device), y.to(device), w.to(device), m.to(device), t_mask.to(device)
                    logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
                    
                    prob_presence = torch.sigmoid(logit_presence.squeeze(1))
                    safe_pred = torch.clamp(pred_abundance.squeeze(1), min=0.0, max=20.0)
                    pred_kg = prob_presence * torch.expm1(safe_pred)
                    true_kg = torch.expm1(y)
                    
                    sum_w = w.sum()
                    if sum_w > 0:
                        abs_error = torch.abs(pred_kg - true_kg)
                        running_mae += ((abs_error * w).sum() / sum_w).item()
                        count += 1
                        
            val_mae = running_mae / max(1, count)
            
            # Enregistrement des metriques dans MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_loss_cls", train_cls, step=epoch)
            mlflow.log_metric("train_loss_reg", train_reg, step=epoch)
            mlflow.log_metric("val_mae", val_mae, step=epoch)
            
            if val_mae < best_local_mae:
                best_local_mae = val_mae
                if val_mae < global_best_mae:
                    global_best_mae = val_mae
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    mlflow.log_artifact(BEST_MODEL_PATH)

            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_local_mae

if __name__ == "__main__":
    print("Initialisation Optimisation")
    
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=MAX_EPOCHS, reduction_factor=3)
    sampler = TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name="ziln_cluster_optim", 
        direction="minimize", 
        storage=DB_NAME, 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Nombre d'essais dans la base : {len(study.trials)}")
    print(f"Lancement de {N_TRIALS} nouveaux essais")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        print("Arret manuel detecte.")
    
    print("OPTIMISATION TERMINEE")
    if len(study.trials) > 0:
        trial = study.best_trial
        print(f"Meilleure MAE Validation : {trial.value:,.0f} kg")
        print("Hyperparametres optimaux :")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
            
        df_results = study.trials_dataframe()
        df_results.to_csv(CSV_RESULTS_PATH, index=False)
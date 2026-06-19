# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:58:33 2026

@author: jdanielou
"""
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import mlflow
import xarray as xr
from prepare_data_cluster import load_and_prepare_catch_data
from dataloader_cluster import MILFisheryDataset
from model_2heads_cluster import WeeklyMILModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Appareil de calcul detecte : {device}", flush=True)

BASE_DATA_DIR = "/marbec-data/Osmose-Montpellier-Vol3/data-jules/data"
PATH_CSV = os.path.join(BASE_DATA_DIR, "input/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv")
PATH_PHYS = os.path.join(BASE_DATA_DIR, "input/phy/")
PATH_CHL = os.path.join(BASE_DATA_DIR, "input/bio/")
PATH_MASK = os.path.join(BASE_DATA_DIR, "input/official_mask_zee.nc")
DOSSIER_MODELS = os.path.join(BASE_DATA_DIR, "output/models/")
MLFLOW_DIR = os.path.join(BASE_DATA_DIR, "output/runs/mlflow")

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']
MAX_EPOCHS = 30
BATCH_SIZE = 128
N_TRIALS = 100

DB_NAME = f"sqlite:///{os.path.join(DOSSIER_MODELS, 'optuna_finale_v4.db')}"
BEST_MODEL_PATH = os.path.join(DOSSIER_MODELS, "meilleur_modele_optuna_ziln_V4.pth")
CSV_RESULTS_PATH = os.path.join(DOSSIER_MODELS, "resultats_optuna_ziln_V4.csv")

os.makedirs(DOSSIER_MODELS, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
mlflow.set_experiment("squid_prediction_ziln_v4")

print("Chargement des donnees CSV...", flush=True)
df_all = load_and_prepare_catch_data(PATH_CSV)

df_train = df_all[(df_all['year'] >= 2012) & (df_all['year'] <= 2023)].reset_index(drop=True)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)

print(f"Taille entrainement : {len(df_train)} evenements", flush=True)
print(f"Taille validation : {len(df_val)} evenements", flush=True)

print("Chargement et fusion des fichiers NetCDF en RAM (cette etape peut prendre quelques minutes)...", flush=True)
files_phys = sorted(glob.glob(os.path.join(PATH_PHYS, "*.nc")))
files_chl = sorted(glob.glob(os.path.join(PATH_CHL, "*.nc")))

ds_phys = xr.open_mfdataset(files_phys, combine='by_coords', engine='h5netcdf')
ds_chl = xr.open_mfdataset(files_chl, combine='by_coords', engine='h5netcdf')

shared_ds = xr.merge([ds_phys, ds_chl])
shared_ds = shared_ds.sel(latitude=slice(-38.0, 3.0), longitude=slice(227.0, 293.0))
shared_ds.load()
print("Chargement des donnees NetCDF termine.", flush=True)

train_ds = MILFisheryDataset(shared_ds, df_train, FEATURES, PATH_MASK, is_train=True)
val_ds = MILFisheryDataset(shared_ds, df_val, FEATURES, PATH_MASK, is_train=False)

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
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    alpha_reg = trial.suggest_float("alpha_reg", 0.5, 5.0)
    delta_huber = trial.suggest_float("delta_huber", 0.5, 3.0)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 4)
    num_fc_layers = trial.suggest_int("num_fc_layers", 1, 3)

    current_batch_size = 64

    print(f"\n--- DEBUT ESSAI {trial.number} ---", flush=True)
    print(f"Arch: dim={hidden_dim}, conv={num_conv_layers}, fc={num_fc_layers} | Batch={current_batch_size}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=current_batch_size, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=current_batch_size, shuffle=False, num_workers=5, pin_memory=True)

    model = WeeklyMILModel(
        in_channels=len(FEATURES), 
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        num_fc_layers=num_fc_layers
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ZeroInflatedLoss(alpha_reg=alpha_reg, delta_huber=delta_huber)
    scaler = torch.amp.GradScaler('cuda')

    best_local_mae = float('inf')

    try:
        global_best_mae = trial.study.best_value
    except ValueError:
        global_best_mae = float('inf')

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)

        for epoch in range(MAX_EPOCHS):
            model.train()
            total_loss, total_cls, total_reg, steps = 0.0, 0.0, 0.0, 0
            
            for x, y, w, m, t_mask in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                t_mask = t_mask.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
                    loss_cls, loss_reg = criterion(logit_presence.squeeze(1), pred_abundance.squeeze(1), y, w)
                    loss = loss_cls + loss_reg
                
                if w.sum() > 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    total_cls += loss_cls.item()
                    total_reg += loss_reg.item()
                    steps += 1

            train_loss = total_loss / max(1, steps)
            train_cls = total_cls / max(1, steps)
            train_reg = total_reg / max(1, steps)

            model.eval()
            running_mae = 0.0
            running_val_loss = 0.0
            count = 0
            with torch.no_grad():
                for x, y, w, m, t_mask in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    w = w.to(device, non_blocking=True)
                    m = m.to(device, non_blocking=True)
                    t_mask = t_mask.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
                        loss_cls, loss_reg = criterion(logit_presence.squeeze(1), pred_abundance.squeeze(1), y, w)
                        val_loss_batch = loss_cls + loss_reg
                        
                        prob_presence = torch.sigmoid(logit_presence.squeeze(1))
                        safe_pred = torch.clamp(pred_abundance.squeeze(1), min=0.0, max=20.0)
                        pred_kg = prob_presence * torch.expm1(safe_pred)
                        true_kg = torch.expm1(y)
                        
                        sum_w = w.sum()
                        if sum_w > 0:
                            running_val_loss += val_loss_batch.item()
                            abs_error = torch.abs(pred_kg - true_kg)
                            running_mae += ((abs_error * w).sum() / sum_w).item()
                            count += 1
                        
            val_mae = running_mae / max(1, count)
            val_loss = running_val_loss / max(1, count)
            
            print(f"  Essai {trial.number} | Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:,.0f} kg", flush=True)
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_loss_cls", train_cls, step=epoch)
            mlflow.log_metric("train_loss_reg", train_reg, step=epoch)
            mlflow.log_metric("val_mae", val_mae, step=epoch)
            
            if val_mae < best_local_mae:
                best_local_mae = val_mae
                if val_mae < global_best_mae:
                    global_best_mae = val_mae
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print("  >>> Nouveau record global. Modele sauvegarde.", flush=True)

            trial.report(val_mae, epoch)
            if trial.should_prune():
                print(f"  --- Essai {trial.number} elague par Optuna.", flush=True)
                raise optuna.exceptions.TrialPruned()

    return best_local_mae

if __name__ == "__main__":
    print("Initialisation Optimisation", flush=True)
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=12, interval_steps=2)
    sampler = TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name="optim_finale_v4", 
        direction="minimize", 
        storage=DB_NAME, 
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler
    )
    
    print(f"Nombre d'essais en base : {len(study.trials)}", flush=True)
    print(f"Lancement de {N_TRIALS} essais", flush=True)
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nArret manuel.", flush=True)
    
    print("\nOPTIMISATION TERMINEE", flush=True)
    if len(study.trials) > 0:
        trial = study.best_trial
        print(f"Meilleure MAE Validation : {trial.value:,.0f} kg", flush=True)
        print("Hyperparametres optimaux :", flush=True)
        for key, value in trial.params.items():
            print(f"  {key}: {value}", flush=True)
            
        df_results = study.trials_dataframe()
        df_results.to_csv(CSV_RESULTS_PATH, index=False)
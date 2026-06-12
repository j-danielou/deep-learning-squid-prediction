# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:34:56 2026

@author: jdanielou
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model_2heads import WeeklyMILModel

# Configuration
device = torch.device("cpu")
print(f"Appareil de calcul : {device}")

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']
MAX_EPOCHS = 8
BATCH_SIZE = 16

# Chargement et Sous-échantillonnage aléatoire
print("Chargement des donnees reelles...")
df_all = load_and_prepare_catch_data(PATH_CSV)

df_train_full = df_all[(df_all['year'] >= 2012) & (df_all['year'] <= 2023)]
df_val_full = df_all[df_all['year'] == 2024]

# Echantillonnage : 500 points pour l'entrainement, 150 pour la validation
df_train = df_train_full.sample(n=500, random_state=42).reset_index(drop=True)
df_val = df_val_full.sample(n=150, random_state=42).reset_index(drop=True)

print(f"Echantillon d'entrainement : {len(df_train)} evenements")
print(f"Echantillon de validation : {len(df_val)} evenements")

train_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_train, FEATURES, PATH_MASK, is_train=True)
val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

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
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    alpha_reg = trial.suggest_float("alpha_reg", 0.1, 5.0)
    delta_huber = trial.suggest_float("delta_huber", 0.5, 3.0)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ZeroInflatedLoss(alpha_reg=alpha_reg, delta_huber=delta_huber)

    best_val_mae = float('inf')

    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y, w, m, t_mask in train_loader:
            x, y, w, m, t_mask = x.to(device), y.to(device), w.to(device), m.to(device), t_mask.to(device)
            optimizer.zero_grad()
            
            logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
            loss_cls, loss_reg = criterion(logit_presence.squeeze(1), pred_abundance.squeeze(1), y, w)
            loss = loss_cls + loss_reg
            
            if w.sum() > 0:
                loss.backward()
                optimizer.step()

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
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae

        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_mae

if __name__ == "__main__":
    print("Demarrage de l'optimisation CPU avec sous-echantillonnage")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="ZILN_RealData_Opt")
    
    try:
        study.optimize(objective, n_trials=20)
        
        print("\nRESULATS")
        trial = study.best_trial
        print(f"Meilleure MAE Validation : {trial.value:,.0f} kg")
        for key, value in trial.params.items():
            print(f"{key}: {value}")
            
    except KeyboardInterrupt:
        print("\nOptimisation interrompue. Meilleur resultat partiel :")
        print(study.best_trial.params)
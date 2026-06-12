# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:04:25 2026

@author: jdanielou
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm

from prepare_data import load_and_prepare_catch_data
from dataloader import MILFisheryDataset
from model_2heads import WeeklyMILModel

#ARCHITECTURE_NAME = "conv3d_transmil_hybride"
ARCHITECTURE_NAME = "swinlstm_transmil_hybride"

class ZeroInflatedLoss(nn.Module):
    """
    Sépare le problème en deux : 
    1. Pénalise les erreurs sur la présence/absence d'habitat (BCE).
    2. Pénalise l'erreur sur le tonnage UNIQUEMENT là où il y a des calmars (Huber).
    """
    def __init__(self, alpha_reg=1.0):
        super(ZeroInflatedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.huber = nn.HuberLoss(reduction='none', delta=1.5)
        self.alpha = alpha_reg

    def forward(self, logit_presence, pred_abundance, y_true, w):
        #tete présence (0 ou 1)
        is_present = (y_true > 0).float()
        loss_cls = self.bce(logit_presence, is_present)
        
        #tete abondance (On ne calcule l'erreur que sur les captures > 0)
        loss_reg = self.huber(pred_abundance, y_true)
        loss_reg = loss_reg * is_present  #annulation gradient océan vide
        
        #ponderation mask
        sum_w = w.sum() + 1e-8
        final_loss_cls = (loss_cls * w).sum() / sum_w
        final_loss_reg = (loss_reg * w).sum() / sum_w
        
        return final_loss_cls, final_loss_reg * self.alpha

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil de calcul : {device}")

PATH_CSV = "D:/deep-learning-squid-prediction/data/v1960_2026-05-05_fishing-activity-monthly-catch-1x1.csv"
PATH_PHYS = "D:/deep-learning-squid-prediction/data/tensors_annuels/phy/"
PATH_CHL = "D:/deep-learning-squid-prediction/data/tensors_annuels/bio/"
PATH_MASK = "D:/deep-learning-squid-prediction/data/official_mask_zee.nc"

FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad', 'fishing_hours']

print("Initialisation du pipeline de donnees")
df_all = load_and_prepare_catch_data(PATH_CSV)

# Entrainement sur la decennie complete
df_train = df_all[df_all['year'] <= 2023].reset_index(drop=True)
df_val = df_all[df_all['year'] == 2024].reset_index(drop=True)
print(f"Taille entrainement (2012-2023) : {len(df_train)} evenements")
print(f"Taille validation (2024) : {len(df_val)} evenements")

train_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_train, FEATURES, PATH_MASK, is_train=True)
val_ds = MILFisheryDataset(PATH_PHYS, PATH_CHL, df_val, FEATURES, PATH_MASK, is_train=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

model = WeeklyMILModel(in_channels=len(FEATURES), hidden_dim=512).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.0015)
criterion = ZeroInflatedLoss(alpha_reg=1.6)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.4,
    patience=4,
    threshold=500.0,
    threshold_mode='abs'
)

def run_epoch(model, loader, optimizer=None):
    if optimizer: 
        model.train()
    else: 
        model.eval()
    
    running_mae_kg = 0.0
    count = 0
    
    desc = "Entrainement" if optimizer else "Validation"
    pbar = tqdm(loader, desc=desc, leave=False)
    
    for x, y, w, m, t_mask in pbar:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)
        m = m.to(device)
        t_mask = t_mask.to(device)
        
        if optimizer: 
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(optimizer is not None):
            #recuperation 2 tetes
            logit_presence, pred_abundance, _ = model(x, pixel_mask=m, time_mask=t_mask)
            logit_presence = logit_presence.squeeze(1)
            pred_abundance = pred_abundance.squeeze(1)
            
            #calcul loss
            loss_cls, loss_reg = criterion(logit_presence, pred_abundance, y, w)
            loss = loss_cls + loss_reg
            sum_weights = w.sum()
            
            if sum_weights > 0:
                if optimizer:
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    #prediction phy : Probabilité * Quantité
                    prob_presence = torch.sigmoid(logit_presence)
                    safe_pred = torch.clamp(pred_abundance, min=0.0, max=20.0)
                    
                    #abondance potentiel par proba habitat bon
                    pred_kg = prob_presence * torch.expm1(safe_pred)
                    true_kg = torch.expm1(y)
                    
                    abs_error = torch.abs(pred_kg - true_kg)
                    mae_kg = (abs_error * w).sum() / sum_weights
                    
                running_mae_kg += mae_kg.item()
                count += 1
                pbar.set_postfix({'MAE (kg)': mae_kg.item()})
                
    return running_mae_kg / max(1, count)


if __name__ == "__main__":
    DOSSIER_MODELS = "D:/deep-learning-squid-prediction/models"
    DOSSIER_RUNS = os.path.join(DOSSIER_MODELS, "runs")
    
    experiment_name = f"squid_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(DOSSIER_RUNS, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    print(f"\nDemarrage. Logs TensorBoard : {log_dir}")

    best_val_loss = float('inf')
    num_epochs = 50

    try:
        for epoch in range(num_epochs):
            train_loss = run_epoch(model, train_loader, optimizer)
            val_loss = run_epoch(model, val_loader)
            
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {curr_lr:.6f}")
            
            writer.add_scalars('Loss', {'Entrainement': train_loss, 'Validation': val_loss}, epoch)
            writer.add_scalar('Hyperparametres/Learning_Rate', curr_lr, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"model_{ARCHITECTURE_NAME}_{timestamp}_mae{int(val_loss)}.pth"
                
                save_path = os.path.join(DOSSIER_MODELS, save_filename)
                torch.save(model.state_dict(), save_path)
                writer.add_text('Info_Modele', f'Nouveau record epoque {epoch+1} sauvegarde sous {save_filename}', epoch)
                
    except KeyboardInterrupt:
        print("\nEntrainement interrompu par l'utilisateur.")
        
    finally:
        writer.close()
        print(f"Processus termine. Les modeles ont ete sauvegardes dans : {DOSSIER_MODELS}")
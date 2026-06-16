# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:07:38 2026

@author: jdanielou
"""
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import torch
import glob
import os
import calendar
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MILFisheryDataset(Dataset):
    def __init__(self, path_phys, path_chl, df_catch, features, mask_path, is_train=False):
        self.files_phys = sorted(glob.glob(os.path.join(path_phys, "*.nc")))
        self.files_chl = sorted(glob.glob(os.path.join(path_chl, "*.nc")))
        
        ds_phys = xr.open_mfdataset(self.files_phys, combine='by_coords', engine='h5netcdf')
        ds_chl = xr.open_mfdataset(self.files_chl, combine='by_coords', engine='h5netcdf')
        
        self.ds = xr.merge([ds_phys, ds_chl])
        self.ds = self.ds.sel(latitude=slice(-38.0, 3.0), longitude=slice(227.0, 293.0))
        
        self.ds.load()
        
        self.features = features
        self.df_catch = df_catch
        self.is_train = is_train
        
        self.gfw_idx = self.features.index('fishing_hours') if 'fishing_hours' in self.features else -1
        
        mask_ds = xr.open_dataset(mask_path)
        self.mask_da = mask_ds['ocean_mask'].load()
        mask_ds.close()

    def __len__(self):
        return len(self.df_catch)
    
    def __getitem__(self, idx):
        row = self.df_catch.iloc[idx]
        lon_360 = row.lon_360
        year, month = int(row.year), int(row.month)
        
        _, num_days = calendar.monthrange(year, month)
        
        ds_month = self.ds.sel(time=f"{year}-{month:02d}")
        lat_slice = slice(row.lat - 0.01, row.lat + 0.99)
        lon_slice = slice(lon_360 - 0.01, lon_360 + 0.99)
        ds_block = ds_month.sel(latitude=lat_slice, longitude=lon_slice)
        
        patch_mask = self.mask_da.sel(latitude=lat_slice, longitude=lon_slice).values
        patch_weight = 1.0 if np.mean(patch_mask) > 0.5 else 0.0
        
        x_list = []
        for feat in self.features:
            val = ds_block[feat].values
            
            if feat == 'CHL':
                val = np.log1p(np.maximum(val, 0.0))
                
            m, s = np.nanmean(val), np.nanstd(val)
            val = (val - m) / (s + 1e-8)
            val = np.nan_to_num(val, nan=0.0)
            
            padded_val = np.zeros((31, val.shape[1], val.shape[2]), dtype=np.float32)
            padded_val[:num_days, :, :] = val
            x_list.append(padded_val)
            
        x_tensor = torch.tensor(np.stack(x_list), dtype=torch.float32)
        
        #Dropout
        if self.is_train and self.gfw_idx != -1:
            if torch.rand(1).item() < 0.15: 
                x_tensor[self.gfw_idx, ...] = 0.0
                
        time_mask = np.zeros(31, dtype=np.float32)
        time_mask[:num_days] = 1.0
        time_mask_tensor = torch.tensor(time_mask, dtype=torch.float32)
                
        y_tensor = torch.tensor(np.log1p(row.harvest_kg), dtype=torch.float32)
        weight_tensor = torch.tensor(patch_weight, dtype=torch.float32)
        mask_tensor = torch.tensor(patch_mask, dtype=torch.float32)
            
        return x_tensor, y_tensor, weight_tensor, mask_tensor, time_mask_tensor
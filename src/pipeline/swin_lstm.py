# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:24:34 2026

@author: jdanielou
"""
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_dim, 
            out_channels=4 * hidden_dim, 
            kernel_size=kernel_size, 
            padding=padding
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        
        i_gate, f_gate, o_gate, c_gate = gates.chunk(4, dim=1)
        
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        c_tilde = torch.tanh(c_gate)
        
        c_next = f * c_prev + i * c_tilde
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class SwinLSTMEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(SwinLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.spatial_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.lstm_cell = ConvLSTMCell(in_channels=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x, time_mask=None):
        b, c, t, h, w = x.size()
        device = x.device
        
        h_t = torch.zeros(b, self.hidden_dim, h, w, device=device)
        c_t = torch.zeros(b, self.hidden_dim, h, w, device=device)
        
        h_states = []
        
        for step in range(t):
            x_t = x[:, :, step, :, :]
            
            if time_mask is not None:
                mask_t = time_mask[:, step].view(b, 1, 1, 1)
                x_t = x_t * mask_t
                
            spatial_features = self.spatial_feature_extractor(x_t)
            h_t, c_t = self.lstm_cell(spatial_features, h_t, c_t)
            
            h_states.append(h_t)
            
        #output shape: [Batch, Hidden, 4_semaines, Height, Width]
        return torch.stack(h_states, dim=2)
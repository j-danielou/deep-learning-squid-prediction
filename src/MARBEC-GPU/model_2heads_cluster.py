# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:09:14 2026

@author: jdanielou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_lstm_cluster import SwinLSTMEncoder

class OceanographicTemporalPool(nn.Module):
    def __init__(self):
        super(OceanographicTemporalPool, self).__init__()

    def masked_mean(self, x, time_mask, start, end):
        mask_expanded = time_mask.view(x.size(0), 1, x.size(2), 1, 1)
        x_slice = x[:, :, start:end, :, :]
        mask_slice = mask_expanded[:, :, start:end, :, :]
        masked_x = x_slice * mask_slice
        sum_x = torch.sum(masked_x, dim=2, keepdim=True)
        valid_days = torch.sum(mask_slice, dim=2, keepdim=True)
        return sum_x / (valid_days + 1e-8)

    def forward(self, x, time_mask):
        p1 = self.masked_mean(x, time_mask, 0, 8)  
        p2 = self.masked_mean(x, time_mask, 8, 16) 
        p3 = self.masked_mean(x, time_mask, 16, 24)
        p4 = self.masked_mean(x, time_mask, 24, 31)
        return torch.cat([p1, p2, p3, p4], dim=2)

class WeeklyMILModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_conv_layers=2, num_fc_layers=1):
        super(WeeklyMILModel, self).__init__()

        self.temporal_pool = OceanographicTemporalPool()
        self.encoder = SwinLSTMEncoder(in_channels=in_channels, hidden_dim=hidden_dim, num_conv_layers=num_conv_layers)
        
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        presence_layers = []
        abundance_layers = []
        curr_dim = hidden_dim
        
        for _ in range(num_fc_layers - 1):
            presence_layers.extend([nn.Linear(curr_dim, curr_dim // 2), nn.GELU(), nn.Dropout(0.2)])
            abundance_layers.extend([nn.Linear(curr_dim, curr_dim // 2), nn.GELU(), nn.Dropout(0.2)])
            curr_dim = curr_dim // 2
            
        presence_layers.append(nn.Linear(curr_dim, 1))
        abundance_layers.append(nn.Linear(curr_dim, 1))
        
        self.head_presence = nn.Sequential(*presence_layers)
        self.head_abundance = nn.Sequential(*abundance_layers)

    def forward(self, x, pixel_mask=None, time_mask=None):
        weekly_x = self.temporal_pool(x, time_mask)
        h_t = self.encoder(weekly_x, time_mask=None)
        
        b, c, t_dim, h, w = h_t.size()
        mil_instances = h_t.view(b, c, -1).transpose(1, 2)

        if pixel_mask is not None:
            mask_expanded = pixel_mask.unsqueeze(1).expand(-1, t_dim, -1, -1).reshape(b, -1)
            key_padding_mask = (mask_expanded == 0)
            
            all_masked = key_padding_mask.all(dim=1)
            key_padding_mask[all_masked, 0] = False
        else:
            key_padding_mask = None

        attn_out, _ = self.self_attention(
            mil_instances, mil_instances, mil_instances, 
            key_padding_mask=key_padding_mask
        )
        x_seq = self.norm1(mil_instances + attn_out)
        
        att_logits = self.attention_pool(x_seq)
        
        if key_padding_mask is not None:
            att_logits = att_logits.masked_fill(key_padding_mask.unsqueeze(-1), -10000.0)
            
        att_weights = torch.softmax(att_logits, dim=1)
        context_vector = torch.sum(x_seq * att_weights, dim=1)
        
        logit_presence = self.head_presence(context_vector)
        pred_abundance = self.head_abundance(context_vector)

        return logit_presence, pred_abundance, att_weights
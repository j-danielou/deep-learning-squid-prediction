# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:11:15 2026

@author: jdanielou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_lstm import SwinLSTMEncoder

class OceanographicTemporalPool(nn.Module):
    """
    Comprime les 31 jours du mois en 4 semaines synoptiques 
    tout en respectant le masque des jours valides.
    """
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
        #x shape : [Batch, Channels, 31, Height, Width]
        p1 = self.masked_mean(x, time_mask, 0, 8)  
        p2 = self.masked_mean(x, time_mask, 8, 16) 
        p3 = self.masked_mean(x, time_mask, 16, 24)
        p4 = self.masked_mean(x, time_mask, 24, 31)
        #Retourne: [Batch, Channels, 4, Height, Width]
        return torch.cat([p1, p2, p3, p4], dim=2)


class WeeklyMILModel(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(WeeklyMILModel, self).__init__()

        self.temporal_pool = OceanographicTemporalPool()
        
        #SwinLSTM
        self.encoder = SwinLSTMEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        
        #TransMIL
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        #Prédiction
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, pixel_mask=None, time_mask=None):
        weekly_x = self.temporal_pool(x, time_mask)
        
        #h_t: [Batch, Hidden, 4, Height, Width]
        h_t = self.encoder(weekly_x, time_mask=None)
        
        b, c, t_dim, h, w = h_t.size()
        
        #mil_instances 576 pixels (4 * 12 * 12)
        mil_instances = h_t.view(b, c, -1).transpose(1, 2)

        #gestion mask
        if pixel_mask is not None:
            mask_expanded = pixel_mask.unsqueeze(1).expand(-1, t_dim, -1, -1).reshape(b, -1)
            key_padding_mask = (mask_expanded == 0)
            
            all_masked = key_padding_mask.all(dim=1)
            key_padding_mask[all_masked, 0] = False
        else:
            key_padding_mask = None

        #Self-Attention
        attn_out, _ = self.self_attention(
            mil_instances, mil_instances, mil_instances, 
            key_padding_mask=key_padding_mask
        )
        x_seq = self.norm1(mil_instances + attn_out)
        
        #Softmax
        att_logits = self.attention_pool(x_seq)
        
        if key_padding_mask is not None:
            att_logits = att_logits.masked_fill(key_padding_mask.unsqueeze(-1), -1e9)
            
        att_weights = torch.softmax(att_logits, dim=1)

        #Agrégation finale et Prédiction
        context_vector = torch.sum(x_seq * att_weights, dim=1)
        y_pred = self.regressor(context_vector)

        return y_pred, att_weights
    
# class WeeklyMILModel(nn.Module):
#     def __init__(self, in_channels, hidden_dim):
#         super(WeeklyMILModel, self).__init__()
#         self.encoder = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
#         self.temporal_pool = OceanographicTemporalPool()
        
#         self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
#         self.norm1 = nn.LayerNorm(hidden_dim)
        
#         self.attention_pool = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim // 2, 1)
#         )
        
#         self.regressor = nn.Linear(hidden_dim, 1)

#     def forward(self, x, pixel_mask, time_mask):
#         features = self.encoder(x)
#         features = F.relu(features)

#         weekly_features = self.temporal_pool(features, time_mask)

#         b, c, t, h, w = weekly_features.size()
        
#         mil_instances = weekly_features.view(b, c, -1).transpose(1, 2)

#         if pixel_mask is not None:
#             mask_expanded = pixel_mask.unsqueeze(1).expand(-1, t, -1, -1).reshape(b, 576)
#             key_padding_mask = (mask_expanded == 0)
            
#             all_masked = key_padding_mask.all(dim=1)
#             key_padding_mask[all_masked, 0] = False
            
#         else:
#             key_padding_mask = None

#         attn_out, _ = self.self_attention(
#             mil_instances, mil_instances, mil_instances, 
#             key_padding_mask=key_padding_mask
#         )
#         x_seq = self.norm1(mil_instances + attn_out)
        
#         att_logits = self.attention_pool(x_seq)
        
#         if key_padding_mask is not None:
#             att_logits = att_logits.masked_fill(key_padding_mask.unsqueeze(-1), -1e9)
            
#         att_weights = torch.softmax(att_logits, dim=1)

#         context_vector = torch.sum(x_seq * att_weights, dim=1)
#         y_pred = self.regressor(context_vector)

#         return y_pred, att_weights
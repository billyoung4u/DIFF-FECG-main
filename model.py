#原始去噪网络
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln

Linear = nn.Linear




class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level=noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

   
    
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.diffusion_projection = Linear(64, in_channels)
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        noise_embed =  self.diffusion_projection(noise_embed)#.unsqueeze(-1)
        batch = x.shape[0]
        noise_embed = noise_embed.expand(batch, -1)
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
          #  print(gamma.shape, noise_embed.shape, x.shape)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x
  
class HNFBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        padding_mode = 'zeros'#'reflect'
        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size//4, 3, dilation=dilation, padding=1*dilation, padding_mode=padding_mode),
            Conv1d(hidden_size, hidden_size//4, 5, dilation=dilation, padding=2*dilation, padding_mode=padding_mode),
            Conv1d(hidden_size, hidden_size//4, 9, dilation=dilation, padding=4*dilation, padding_mode=padding_mode),
            Conv1d(hidden_size, hidden_size//4, 15, dilation=dilation, padding=7*dilation, padding_mode=padding_mode),
        ])
        
        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode=padding_mode)
        
        self.norm =nn.InstanceNorm1d(hidden_size//2)
        
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode=padding_mode)
        
    def forward(self, x):
        residual = x
        
        filts = []
        for layer in self.filters:
            filts.append(layer(x))
            
        filts = torch.cat(filts, dim=1)
        
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.5)
    
        
        filts = F.leaky_relu(self.conv_2(filts), 0.5)

        return filts + residual

class DiffBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        

        self.dilated_conv = Conv1d(hidden_size, 2*hidden_size, 3, padding=dilation, dilation=dilation)
        self.output_residual = Conv1d(hidden_size, hidden_size, 1)
    def forward(self, x,con):
       # con = self.hnf(con)
        y = self.dilated_conv(con)
     #   print(con.shape, y.shape)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = y+x
     #   print(y.shape)
        y  = self.output_residual(y)
        return y

       
    
    
class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        padding_mode = 'zeros'
        self.encoding = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, padding_mode=padding_mode)
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, padding_mode=padding_mode)
    
    def forward(self, x, noise_embed):
        x = self.input_conv(x)

        x = self.encoding(x, noise_embed)
        return self.output_conv(x)
    

class ConditionalModel(nn.Module):
    def __init__(self, feats=64):
        super(ConditionalModel, self).__init__()

        padding_mode = 'zeros'
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode=padding_mode),
                          ),#nn.LeakyReLU(0.2)
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        
        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode=padding_mode),
                          ),#nn.LeakyReLU(0.2)
            DiffBlock(feats, feats, 1),
            DiffBlock(feats, feats, 2),
            DiffBlock(feats, feats, 4),
            DiffBlock(feats, feats, 2),
            DiffBlock(feats, feats, 1),
        ])
        
        self.embed = PositionalEncoding(64)#feats
        
        self.bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])
        
        self.conv_out = Conv1d(feats, 1,1)
        
    def forward(self, x, cond, noise_scale):

        x = x.unsqueeze(1)
        cond = cond.unsqueeze(1)


        noise_embed = self.embed(noise_scale)

        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed)) 
        idx = 0
        for x, layer in zip(xs, self.stream_cond):
            if idx == 0:
                cond = layer(cond)+x
            else:
                cond = layer(x, cond)
            idx+= 1
        return self.conv_out(cond)


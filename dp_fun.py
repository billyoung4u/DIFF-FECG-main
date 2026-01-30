from torch.nn import init
import torch, time, os, shutil
import  util
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import FECGDataset
from config import cfg
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import torch.nn.functional as F
import data_util as du
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1 and classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir,model_filename):
 #   current_w = os.path.join(model_save_dir, c.current_w)
    best_w = os.path.join(model_save_dir,model_filename)
   # torch.save(state, current_w)
    torch.save(state, best_w)
   # if is_best: shutil.copyfile(current_w, best_w)



def load_model(c,name, model_save_dir,model_filename):
    model = get_model(c,name)
    best_w = os.path.join(model_save_dir, model_filename)
    state = torch.load(best_w, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model.cuda()
   # print('train with pretrained weight val_loss', state['loss'])
    return model



def exclude_ann_err(ref, ans,fs):
    on = 185*fs #-10*fs
    off = 212*fs# -10*fs
    new_ref = []
    new_ans = []
    for i in range(len(ref)):
        if ref[i] >= on and ref[i]<=off:
            continue
        new_ref.append(ref[i])
    for i in range(len(ans)):
        if ans[i] >= on and ans[i]<=off:
            continue
        new_ans.append(ans[i])
    return new_ref, new_ans








def train_epoch(c,model, optimizer, criterion, scheduler, train_dataloader, show_interval=100, show_bar = False):
    model.train()

    losses = []
    total = 0
    tbar = tqdm(train_dataloader, disable = not show_bar)
    for i, (inputs, target) in enumerate(tbar):
       # print(inputs.shape, target.shape)
        data = inputs.to(device)
        data = data.to(torch.float32)
        labelt = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
    #    print(output.shape, labelt.shape)
        if c.fecg_label is not True:
            output = F.sigmoid(output)
        
        loss = criterion(output,labelt.to(torch.float32)) 
      #  loss = F.mse_loss(Y_est, Y) + F.binary_cross_entropy_with_logits(Y_mask, mask)
        
      #  print(i)
        losses.append(loss.item())
       
        loss.backward()
      #  print(i)
        optimizer.step()
      #  if c.lr_policy == 'plateau':
      #      scheduler.step(loss)
      #  else:
        scheduler.step()
      #  torch.cuda.empty_cache()
      #  print(i)
        
        # print("epoch:{},lossf:{}".format(i,loss.item()))
                
    tbar.close()       
    for i in range(len(losses)):
        total = total + losses[i]
        
    total /= len(losses)
    # print("epoch:{},loss_train:{}".format(show_interval,total))
                
    return total


def val_epoch(c,model, optimizer, criterion, scheduler, val_dataloader, show_interval=100, show_bar = False):
    model.eval()
    losses = []
    total = 0
    tbar = tqdm(val_dataloader, disable =not show_bar)
    for i, (inputs, target) in enumerate(tbar):     
        data = inputs.to(device)
        data = data.to(torch.float32)
        labelt = target.to(device)

        optimizer.zero_grad()
        output = model(data)
     #   Y_est,Y_mask = model(X)
        if c.fecg_label is not True:
            output = F.sigmoid(output)

        

        loss = criterion(output,labelt.to(torch.float32)) 

        losses.append(loss.item())

            
    tbar.close()        
    for i in range(len(losses)):
        total = total + losses[i]
        
    total /= len(losses)
    # print("epoch:{},loss_val:{}".format(show_interval,total))
                 
    return total#,0
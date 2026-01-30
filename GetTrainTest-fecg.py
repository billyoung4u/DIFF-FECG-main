import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
import json
import pandas as pd
from scipy import signal

def get_model():

    # from model import ConditionalModel
    # model = ConditionalModel(feats = 64).cuda()

    from mkf2 import ConditionalModel
    model = ConditionalModel(feats=64).cuda()

    return model

def get_loss():
  sloss1 = nn.MSELoss( reduction='mean')
  sloss2 = nn.MSELoss( reduction='mean')
  def diff_loss(noise1, noise2, waveform,p_waveform,weight = 0.5):
      loss1 = sloss1(noise1, noise2)
      loss2 = sloss2(waveform, p_waveform)

      loss = weight*loss1+ (1-weight)*loss2
      return loss
  return diff_loss




def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffLearner:
  def __init__(self, model_dir, model, dataset, optimizer, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer

    self.autocast = torch.cuda.amp.autocast(enabled=False)
    self.scaler = torch.cuda.amp.GradScaler(enabled=False)
    self.step = 0
    self.is_master = True
    self.noise_schedule  =np.linspace(1e-4, 0.035, 50).tolist()
    beta = np.array(self.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = get_loss()
    self.loss_fn2 = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict, pretrain=False):
 
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):

        self.model.module.load_state_dict(state_dict['model'])
    else:

        self.model.load_state_dict(state_dict['model'])
    
  #  if not pretrain:
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):

    save_name= f'{self.model_dir}/{filename}.pt'
   # print(save_name)
    torch.save(self.state_dict(), save_name)

  def restore_from_checkpoint(self,pretrain_path=None, filename='weights'):
      filename = c.model_name + '_model'
      try:
        checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
        self.load_state_dict(checkpoint)
        return True
      except FileNotFoundError:
        return False

  def train(self, max_steps=None):
    
    device = next(self.model.parameters()).device
    min_loss = 1111100000
    
    while True:
      tbar = tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}', disable = not c.show_bar )

      for features in tbar:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)

        #新增代码：实时显示 Loss
        tbar.set_postfix(loss=f"{loss.item():.6f}", step=self.step)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if min_loss > loss.item():
              min_loss = loss.item()
            #  print('save model',model_file)
              self.save_to_checkpoint(filename = model_file)
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    noisy,waveform=features
    waveform = waveform.squeeze(dim=1)
    waveform = waveform #*0.2
    noisy = noisy.squeeze(dim=1)
    waveform = waveform.to(torch.float32)
    noisy = noisy.to(torch.float32)

    N, T = waveform.shape
    device = waveform.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
        t = torch.randint(0, len(self.noise_schedule), [N], device=waveform.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        m = (((1-self.noise_level[t])/self.noise_level[t]**0.5)**0.5).unsqueeze(1) 
        noise = torch.randn_like(waveform)
        noisy_waveform = (1-m) * noise_scale_sqrt  * waveform + m * noise_scale_sqrt * noisy  + (1.0 - (1+m**2) *noise_scale)**0.5 * noise
        combine_noise = (m * noise_scale_sqrt * (noisy-waveform) + (1.0 - (1+m**2) *noise_scale)**0.5 * noise) / (1-noise_scale)**0.5
        predicted = self.model(noisy_waveform, noisy, t)
        predicted_waveform = 1/(noise_scale**0.5)*(noisy_waveform-(1-noise_scale)**0.5 * predicted.squeeze(1))

        loss = self.loss_fn(combine_noise, predicted.squeeze(1), waveform,predicted_waveform,weight = c.loss_weight)

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss




def _train_impl(replica_id, model, dataset):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=2e-4)
  max_steps = 200*40

  learner = DiffLearner(c.model_save_dir, model, dataset, opt,  fp16=False)
  learner.is_master = True
  learner.train(max_steps=max_steps)


def train():
  #  util.setup_seed(c.seed)
    dataset = FECGDataset(c,db=c.db,train=True, seg_len = c.seg_len, fs = c.fs, 
                            test_idx = test_idx,aecg_channel = c.aecg_channel, smooth = c.smooth,
                            fecg_label=c.fecg_label,mecg_label= False,remove_pl = False,
                           train_all_channel = c.train_all_channel)
    train_dataloader = DataLoader(dataset, batch_size=c.batch_size, shuffle=True, num_workers=0)
    model = get_model()
    _train_impl(0, model, train_dataloader)

    
    
    
models = {}


def load_model(model_dir=None, model_file = None, device=torch.device('cuda')):

    checkpoint = torch.load(f'{model_dir}/{model_file}.pt')

    model = get_model()
    #   print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
    model = models[model_dir]
 
      
    return model
      

def inference_schedule(model):
    noise_schedule = np.linspace(1e-4, 0.035, 50).tolist()
    training_noise_schedule = np.array(noise_schedule)
    inference_noise_schedule = training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)
    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1): 
      sigmas[n] = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    m = [0 for i in alpha] 
    gamma = [0 for i in alpha] 
    delta = [0 for i in alpha]  
    d_x = [0 for i in alpha]  
    d_y = [0 for i in alpha]  
    delta_cond = [0 for i in alpha]  
    delta_bar = [0 for i in alpha] 
    c1 = [0 for i in alpha] 
    c2 = [0 for i in alpha] 
    c3 = [0 for i in alpha] 
    oc1 = [0 for i in alpha] 
    oc3 = [0 for i in alpha] 
    
    for n in range(len(alpha)):
      m[n] = min(((1- alpha_cum[n])/(alpha_cum[n]**0.5)),1)**0.5
    m[-1] = 1    

    for n in range(len(alpha)):
      delta[n] = max(1-(1+m[n]**2)*alpha_cum[n],0)
      gamma[n] = sigmas[n]

    for n in range(len(alpha)):
      if n >0: 
        d_x[n] = (1-m[n])/(1-m[n-1]) * (alpha[n]**0.5)
        d_y[n] = (m[n]-(1-m[n])/(1-m[n-1])*m[n-1])*(alpha_cum[n]**0.5)
        delta_cond[n] = delta[n] - (((1-m[n])/(1-m[n-1])))**2 * alpha[n] * delta[n-1]
        delta_bar[n] = (delta_cond[n])* delta[n-1]/ delta[n]
      else:
        d_x[n] = (1-m[n])* (alpha[n]**0.5)
        d_y[n]= (m[n])*(alpha_cum[n]**0.5)
        delta_cond[n] = 0
        delta_bar[n] = 0 


    for n in range(len(alpha)):
      oc1[n] = 1 / alpha[n]**0.5
      oc3[n] = oc1[n] * beta[n] / (1 - alpha_cum[n])**0.5
      if n >0:
        c1[n] = (1-m[n])/(1-m[n-1])*(delta[n-1]/delta[n])*alpha[n]**0.5 + (1-m[n-1])*(delta_cond[n]/delta[n])/alpha[n]**0.5
        c2[n] = (m[n-1] * delta[n] - (m[n] *(1-m[n]))/(1-m[n-1])*alpha[n]*delta[n-1])*(alpha_cum[n-1]**0.5/delta[n])
        c3[n] = (1-m[n-1])*(delta_cond[n]/delta[n])*(1-alpha_cum[n])**0.5/(alpha[n])**0.5
      else:
        c1[n] = 1 / alpha[n]**0.5
        c3[n] = c1[n] * beta[n] / (1 - alpha_cum[n])**0.5
    return alpha, beta, alpha_cum,sigmas, T, c1, c2, c3, delta, delta_bar
      

def predict( model, noisy_waveform, alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar, device=torch.device('cuda')):
  
  with torch.no_grad():

    waveform = torch.randn(noisy_waveform.shape[0], noisy_waveform.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
        if n > 0:
            predicted_noise =  model(waveform, noisy_waveform, torch.tensor([T[n]], device=waveform.device)).squeeze(1)
            waveform = c1[n] * waveform + c2[n] * noisy_waveform - c3[n] * predicted_noise
            noise = torch.randn_like(waveform)
            newsigma= delta_bar[n]**0.5 
            waveform += newsigma * noise
        else:
            predicted_noise =  model(waveform, noisy_waveform, torch.tensor([T[n]], device=waveform.device)).squeeze(1)
            waveform = c1[n] * waveform - c3[n] * predicted_noise

  return waveform


def test(model_dir):


  
  test_dataset = FECGDataset(c,db=c.db,train=False, seg_len = c.seg_len, fs = c.fs, test_idx = test_idx,
                       aecg_channel = c.aecg_channel, smooth = c.smooth, fecg_label=c.fecg_label,mecg_label= c.mecg_label,remove_pl = False, train_all_channel = c.train_all_channel)
  test_dataloader = DataLoader(test_dataset, batch_size=c.batch_size, num_workers=0)  
    
  model = load_model(model_dir=model_dir , model_file = model_file)
  alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar = inference_schedule(model)



  ori_mecg, ori_fecg,pre_mecg, pre_fecg= None,None,None, None
  for i, (inputs, target) in enumerate(test_dataloader):
    noisy_signal = inputs
    clean_signal = target
    clean_signal = clean_signal# *0.2
    noisy_signal = noisy_signal.squeeze(dim=1)
    clean_signal = clean_signal.squeeze(dim=1)
    noisy_signal = noisy_signal.to(torch.float32).cuda()
    clean_signal = clean_signal.to(torch.float32).cuda()

    wlen = noisy_signal.shape[1]
    singal = None
    for k in range(c.sample_k):
        tmp_singal = predict( model, noisy_signal, alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar)
        tmp_singal = tmp_singal[:,:wlen]
        if singal is None:
            singal = tmp_singal
        else:
            singal += tmp_singal
    singal /= c.sample_k

    if pre_fecg is None:
        ori_fecg = clean_signal.detach().cpu().numpy().flatten()
        pre_fecg = singal.detach().cpu().numpy().flatten()
    else:
        pre_fecg = np.hstack((pre_fecg, singal.detach().cpu().numpy().flatten()))
        ori_fecg = np.hstack((ori_fecg, clean_signal.detach().cpu().numpy().flatten()))
        
  pre_fecg = data_reshape(pre_fecg,c.seg_len)
  ori_fecg = data_reshape(ori_fecg,c.seg_len)
  fqrs_rpeaks = test_dataset.get_fqrs()
  record_len = test_dataset.get_record_len()
    

  ori_fecg = du.butter_bandpass_filter(ori_fecg, c.filter[0], c.filter[1], c.fs, axis = 0)
  pre_fecg = du.butter_bandpass_filter(pre_fecg, c.filter[0], c.filter[1], c.fs, axis = 0)


  pcc,csim,prd,spec_corr, mae, mse= util.evaluate_signal(ori_fecg, pre_fecg)
  return pcc,csim,prd,spec_corr, mae, mse

def data_reshape(ecg,seg_len):
    ecg = ecg.reshape((len(ecg)//c.seg_len,1, c.seg_len))
    ecg = du.deframe(ecg)
    return ecg

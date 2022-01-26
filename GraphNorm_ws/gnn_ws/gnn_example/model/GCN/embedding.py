import torch.nn as nn
import torch
import numpy as np
from model.Norm import *


class GaussianFourierProjectionTime(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)


# Small network that encodes the noise-level or noisy label embedding
class Time_Embedding(nn.Module):

  def __init__(self, num_classes, out_ch, label=False, latent=False, normalize=False, normalize_spectral=False): # use label=True, for the embedding of the noisy label
    super().__init__()
    self.act = act = nn.ELU()
    self.nf = nf = 1
    self.label = label
    self.latent = latent
    self.normalize = normalize

    ## timestep/noise_level embedding
    modules = []

    if not self.label and not self.latent:
      modules.append(GaussianFourierProjectionTime(
        embedding_size=nf, scale=1.0
      ))
      nf_start = nf * 2
    elif self.label:
      nf_start = num_classes
    elif self.latent:
      nf_start = 128

    if normalize_spectral:
      modules.append(torch.nn.utils.spectral_norm(nn.Linear(nf_start, nf * 4)))
    else:
      modules.append(nn.Linear(nf_start, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    if normalize_spectral:
      modules.append(torch.nn.utils.spectral_norm(nn.Linear(nf * 4, out_ch)))
    else:
      modules.append(nn.Linear(nf * 4, out_ch))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    self.all_modules = nn.ModuleList(modules)


  def forward(self, t, print_=False):
    modules = self.all_modules
    m_idx = 0
    if not self.label and not self.latent:
      t = torch.log(t * 999) # t in [0, 999]; lowest to highest noise level (as done in github.com/yang-song/score_sde, makes it in range [-6.907755, 6.906755])
      # Gaussian Fourier features embeddings.
      t = modules[m_idx](t) # Fourier embedding
      m_idx += 1

    
    temb = modules[m_idx](t) # Linear
    m_idx += 1 
    temb = modules[m_idx](self.act(temb)) # Act-Linear
    m_idx += 1

    if self.normalize:
      temb = temb * torch.rsqrt(torch.mean(temb ** 2, dim=1, keepdim=True) + 1e-8) # Normalization layer https://github.com/huangzh13/StyleGAN.pytorch/blob/155a923947b873832689b75e47346ea23e0cbb22/models/CustomLayers.py

    if print_:
      if self.label:
        print("-- label embedding --")
      else:
        print("-- time embedding --")
      #print(f'mean(Bias/Slope): {torch.mean(torch.abs(modules[m_idx].bias/modules[m_idx].weight.data), dim=0)}')
      print(f'Var(emb, dim=mini-batch): {torch.var(temb, unbiased=False, dim=0)}')

    return temb





### Modified from https://github.com/yang-song/score_sde_pytorch

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# Swish and EvoNorm taken from https://github.com/digantamisra98/EvoNorm/blob/master/models/evonorm2d.py
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

class EvoNorm2D(nn.Module):

    def __init__(self, ch, efficient = False, affine = True, eps = 1e-5, groups = 32):
        super(EvoNorm2D, self).__init__()
        self.efficient = efficient
        self.groups = groups
        self.eps = eps
        self.affine = affine
        if self.efficient:
          self.swish = MemoryEfficientSwish()
        else:
          self.v = nn.Parameter(torch.ones(1, ch, 1, 1))
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))
    
    def forward(self, x):
      if not self.efficient:
          num = x * torch.sigmoid(self.v * x) # Original Swish Implementation, however memory intensive.
      else:
          num = self.swish(x) # Memory Efficient Variant of Swish
      if self.affine:
        return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta
      else:
        return num / group_std(x, groups = self.groups, eps = self.eps)

def get_act(opt):
  """Get activation functions from the opt file."""

  if opt.nonlin.lower() == 'elu':
    return nn.ELU()
  elif opt.nonlin.lower() == 'relu':
    return nn.ReLU()
  elif opt.nonlin.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif opt.nonlin.lower() == 'swish':
    return nn.SiLU()
  elif opt.nonlin.lower() == 'mish':
    return nn.Mish()
  else:
    raise NotImplementedError('activation function does not exist!')

def get_norm(norm, ch, affine=True, twod=True):
  """Get activation functions from the opt file."""

  if norm == 'none':
    return nn.Identity()
  elif norm == 'batch':
    if twod:
      return nn.BatchNorm2d(ch, affine = affine)
    else:
      return nn.BatchNorm1d(ch, affine = affine)
  elif norm == 'evo':
    return EvoNorm2D(ch = ch, affine = affine, eps = 1e-5, groups = min(ch // 4, 32))
  elif norm == 'group':
    return nn.GroupNorm(num_groups=min(ch // 4, 32), num_channels=ch, eps=1e-5, affine=affine)
  elif norm == 'layer':
    return nn.LayerNorm(normalized_shape=ch, eps=1e-5, elementwise_affine=affine)
  else:
    raise NotImplementedError('norm choice does not exist')

class get_act_norm(nn.Module):
  def __init__(self, act, norm, ch, emb_dim = None, spectral=False, no_act=False, twod=True):
    super(get_act_norm, self).__init__()
    
    self.norm = norm
    self.act = act
    self.no_act = no_act or self.norm == 'evo' #  we don't apply activation with evo-norm since it's part of it
    
    if emb_dim is not None:
      if spectral:
        self.Dense_0 = torch.nn.utils.spectral_norm(nn.Linear(emb_dim, 2*ch))
      else:
        self.Dense_0 = nn.Linear(emb_dim, 2*ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)
      affine = False # We remove scale/intercept after normalization since we will learn it with [temb, yemb]
    else:
      affine = True

    self.Norm_0 = get_norm(norm, ch, affine, twod)
        
  def forward(self, x, emb = None):
    if emb is not None:
      #emb = torch.cat([temb, yemb], dim=1) # Combine embeddings
      emb_out = self.Dense_0(self.act(emb))#[:, :, None, None] # Linear projection
      # ada-norm as in https://github.com/openai/guided-diffusion
      scale, shift = torch.chunk(emb_out, 2, dim=-1)
      #print(scale.size())
      #print(shift.size())
      #print(x.size())
      x = self.Norm_0(x) * (1 + scale) + shift
    else:
      x = self.Norm_0(x)
    if not self.no_act: #  we don't apply activation with evo-norm since it's part of it
      x = self.act(x)
    return(x)

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)

class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GaussianFourierProjectionTime(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


# Added multi-head attention similar to https://github.com/openai/guided-diffusion/blob/912d5776a64a33e3baf3cff7eb1bcba9d9b9354c/guided_diffusion/unet.py#L361
class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale
    if n_head_channels == -1:
        self.n_heads = n_heads
    else:
        assert channels % n_head_channels == 0
        self.n_heads = channels // n_head_channels

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    C = C // self.n_heads

    w = torch.einsum('bchw,bcij->bhwij', q.reshape(B * self.n_heads, C, H, W), k.reshape(B * self.n_heads, C, H, W)) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B * self.n_heads, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B * self.n_heads, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v.reshape(B * self.n_heads, C, H, W))
    h = h.reshape(B, C * self.n_heads, H, W)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


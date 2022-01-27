import torch
import torch.nn as nn
import numpy as np


def get_norm(norm, ch, affine=True, twod=True):
  """Get activation functions from the opt file."""

  if norm == 'none':
    return nn.Identity()
  elif norm == 'batch':
    if twod:
      return nn.BatchNorm2d(ch, affine = affine)
    else:
      return nn.BatchNorm1d(ch, affine = affine)
  elif norm == 'group':
    return nn.GroupNorm(num_groups=min(ch // 4, 32), num_channels=ch, eps=1e-5, affine=affine)
  elif norm == 'layer':
    return nn.LayerNorm(normalized_shape=ch, eps=1e-5, elementwise_affine=affine)
  else:
    raise NotImplementedError('norm choice does not exist')


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


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


class Norm(nn.Module):

    def __init__(self, norm_type, hidden_dim=64, print_info=None):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
    

class ContNorm(nn.Module):
  def __init__(self, act, norm, ch, emb_dim = None, spectral=False, no_act=False, twod=True):
    super(ContNorm, self).__init__()
    
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
        
  def forward(self, graph, x, emb = None):
    if emb is not None:
      #emb = torch.cat([temb, yemb], dim=1) # Combine embeddings
      #emb_out = self.Dense_0(emb)#[:, :, None, None] # Linear projection
      # ada-norm as in https://github.com/openai/guided-diffusion
      #scale, shift = torch.chunk(emb_out, 2, dim=-1)
      
      node_per_graph = graph.batch_num_nodes
      node_per_graph = torch.Tensor(node_per_graph).long().to(x.device)
      #scale = scale.repeat_interleave(node_per_graph, dim=0)
      shift = emb.repeat_interleave(node_per_graph, dim=0)
      #print(scale.size())
      #print(shift.size())
      #print(x.size())
      x = self.Norm_0(x) #+ shift #self.Norm_0(x) * (1 + scale) + shift #
    else:
      x = self.Norm_0(x)
    if not self.no_act: #  we don't apply activation with evo-norm since it's part of it
      x = self.act(x)
    return(x)
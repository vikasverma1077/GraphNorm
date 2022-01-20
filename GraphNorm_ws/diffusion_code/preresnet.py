import torch
import torch.nn as nn
import torch.nn.functional as F
from sde.sde_lib import VPSDE

from torch.autograd import Variable
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import to_one_hot, mixup_process, get_lambda
from load_data import per_image_standardization
import random

from torchvision.utils import save_image as save_image

import embedding 
import unet_layers
get_act = unet_layers.get_act
default_init = unet_layers.default_init
EvoNorm2D = unet_layers.EvoNorm2D
get_act_norm = unet_layers.get_act_norm
get_norm = unet_layers.get_norm

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, act, norm, in_planes, planes, stride=1, dropout=0, emb_dim=None, spectral=False):
        super(PreActBlock, self).__init__()

        self.actnorm1 = get_act_norm(act, norm, in_planes, emb_dim, spectral)  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.actnorm2 = get_act_norm(act, norm, planes, emb_dim, spectral)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop = nn.Dropout(dropout)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, emb=None, train=True):
        out = self.actnorm1(x, emb)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.actnorm2(out, emb)
        out = self.drop(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, act, norm, in_planes, planes, stride=1, dropout = 0, emb_dim=None, spectral=False):
        super(PreActBottleneck, self).__init__()
        self.actnorm1 = get_act_norm(act, norm, in_planes, emb_dim, spectral)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.actnorm2 = get_act_norm(act, norm, planes, emb_dim, spectral)  
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.actnorm3 = get_act_norm(act, norm, planes, emb_dim, spectral)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, emb=None):
        out = self.actnorm1(x, emb) 
        #out = self.drop(out) # unsure if we should put a dropout here
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.actnorm2(out, emb)
        #out = self.drop(out) # unsure if we should put a dropout here
        out = self.conv2(out)
        out = self.actnorm3(out, emb)
        out = self.drop(out)
        out = self.conv3(out)
        out += shortcut
        return out

class FFN(nn.Module):

    def __init__(self, act, norm, nin, nhid, nout, mlp_type="fc", nlayers=1, act_last=True, emb_dim=None, spectral=False):
        super(FFN, self).__init__()
        self.nlayers = nlayers
        self.mlp_type = mlp_type
        self.act = act
        self.act_last = True
        self.emb_dim = emb_dim

        self.l0 = nn.Linear(nin, nhid)

        layers = []
        for i in range(self.nlayers):
            if self.mlp_type == 'transformer': # norm(lin(act(lin(x))) + x)
                layers.append(nn.ModuleList([nn.Linear(nhid,nhid), nn.Linear(nhid,nhid), get_act_norm(act, norm, nhid, emb_dim = emb_dim, spectral=spectral, no_act=True, twod=False)]))
            elif self.mlp_type == 'resnet': # x + lin(act(norm(x)))
                layers.append(nn.ModuleList([nn.Linear(nhid,nhid), get_act_norm(act, norm, nhid, emb_dim = emb_dim, spectral=spectral, no_act=True, twod=False)]))
            else: # lin(act(norm(x)))
                layers.append(nn.ModuleList([nn.Linear(nhid,nhid), get_act_norm(act, norm, nhid, emb_dim = emb_dim, spectral=spectral, no_act=True, twod=False)]))
        self.layers = nn.ModuleList(layers)

        self.ly = nn.Linear(nhid, nout)

    def forward(self, x, emb=None):
        x = self.l0(x)

        for i in range(self.nlayers):
            current_layer = self.layers[i]
            if self.mlp_type == 'transformer': # norm(lin(act(lin(x))) + x)
                x_ = current_layer[1](self.act(current_layer[0](x)))
                if self.emb_dim is None:
                    x = current_layer[2](x_ + x)
                else:
                    x = current_layer[2](x_ + x, emb)
            elif self.mlp_type == 'resnet': # x + lin(act(norm(x)))
                if self.emb_dim is None:
                    x = x + current_layer[0](self.act(current_layer[1](x)))
                else:
                    x = x + current_layer[0](self.act(current_layer[1](x, emb)))
            else: # lin(act(norm(x)))
                if self.emb_dim is None:
                    x = current_layer[0](self.act(current_layer[1](x)))
                else:
                    x = current_layer[0](self.act(current_layer[1](x, emb)))

        if self.act_last:
            x = self.act(x)
        x = self.ly(x)
        return x

class PreActResNet(nn.Module):
    def __init__(self, args, block, num_blocks, initial_channels, num_classes, per_img_std= False, stride=1):
        super(PreActResNet, self).__init__()

        self.args = args
        self.act = act = get_act(self.args)
        self.norm = norm = self.args.norm
        if self.args.nonlin.lower() == 'relu':
            self.act_end = nn.LeakyReLU()
        else:
            self.act_end = self.act
        self.dropout = self.args.dropout_p

        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        #import pdb; pdb.set_trace()
        self.sde = VPSDE(beta_min=self.args.vp_beta_min, beta_max=self.args.vp_beta_max, N=self.args.diffusion_num_scales)
        if self.args.add_noise_to_image:
            self.sde_img = VPSDE(beta_min=self.args.vp_beta_min_image, beta_max=self.args.vp_beta_max_image, N=self.args.diffusion_num_scales)

        if args.dataset == 'tiny-imagenet-200' or args.dataset == 'shapes':
            data_mult = 4
        else:
            data_mult = 1

        if self.args.score:
            emb_mult = 2  # x2 since we combine temb and yemb
            if self.args.add_noise_to_image_sep:
                emb_mult += 1  # timgemb 
            if self.args.mixup_cond:
                emb_mult += 1  # timgemb 
            if self.args.pred_latent:
                emb_mult += 1

            emb_dim = emb_mult*(self.args.score_nf * 4)
            if self.args.add_noise_to_image_sep:
                self.TimeNet_0_img = embedding.Time_Embedding(self.args, emb_dim // emb_mult, label=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
            if self.args.mixup_cond:
                self.TimeNet_0_lam = embedding.Time_Embedding(self.args, emb_dim // emb_mult, label=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
            self.TimeNet_0 = embedding.Time_Embedding(self.args, emb_dim // emb_mult, label=False, latent=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
            
            # Different types of embeddings for the noisy label
            if self.args.yemb_type_special == 0:
                self.LabelNet_0 = embedding.Time_Embedding(self.args, emb_dim // emb_mult, label=True, latent=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
            else: 
                self.LabelNet_0 = nn.Sequential(nn.Linear(self.args.num_classes, emb_dim // emb_mult, bias=False))

            self.LatentNet_0 = embedding.Time_Embedding(self.args, emb_dim // emb_mult, label=False, latent=True, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)

            if self.args.cond_type == 'concatenate':
                if self.args.add_noise_to_image_sep:
                    self.TimeNet_1_img = embedding.Time_Embedding(self.args, initial_channels, label=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
                if self.args.mixup_cond:
                    self.TimeNet_1_lam = embedding.Time_Embedding(self.args, initial_channels, label=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
                self.TimeNet_1 = embedding.Time_Embedding(self.args, initial_channels, label=False, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
                self.LabelNet_1 = embedding.Time_Embedding(self.args, initial_channels, label=True, normalize=args.embed_norm, normalize_spectral=args.embed_snorm)
                self.c1 = nn.Sequential(nn.Conv2d(initial_channels*3, initial_channels, kernel_size=3, padding=1, bias=True))
                self.c2 = nn.Sequential(nn.Conv2d(initial_channels*(2+1), initial_channels*1, kernel_size=3, padding=1, bias=True))
                self.c3 = nn.Sequential(nn.Conv2d(initial_channels*(2+2), initial_channels*2, kernel_size=3, padding=1, bias=True))
                self.c4 = nn.Sequential(nn.Conv2d(initial_channels*(2+4), initial_channels*4, kernel_size=3, padding=1, bias=True))
            if self.args.cond_type == 'concatenate' or self.args.cond_type == 'embedding_cat':
                mlp_in = emb_dim + initial_channels*8*block.expansion*data_mult
            else: # We use intercept/slope scaling after each normalization
                mlp_in = initial_channels*8*block.expansion*data_mult
        else:
            emb_dim = None
            mlp_in = initial_channels*8*block.expansion*data_mult   

        self.emb_dim = emb_dim

        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)

        if self.args.pred_latent_net_type == "leaky":
            self.z_proj_net = nn.Sequential(nn.Linear(mlp_in, 512), nn.LeakyReLU(), nn.Linear(512, 128), nn.Linear(128, 128))
        else:
            self.z_proj_net = FFN(act=self.act_end, norm=self.norm, nin=mlp_in, nhid=1024, nout=128, mlp_type='fc', nlayers=1, act_last=True, emb_dim=None, spectral=args.ada_snorm)
        self.z_pred_net = nn.Sequential(nn.Linear(128, 128))

        if self.args.pred_noise_net_type == "leaky":
            self.z_pred_net2 = nn.Sequential(nn.Linear(mlp_in, 512), nn.LeakyReLU(), nn.Linear(512, num_classes))
        else:
            self.z_pred_net2 = FFN(act=self.act_end, norm=self.norm, nin=mlp_in, nhid=1024, nout=num_classes, mlp_type='fc', nlayers=1, act_last=True, emb_dim=None, spectral=args.ada_snorm)

        if self.args.mlp_old:
            self.mlp_out = nn.Linear(mlp_in, num_classes)
            self.mlp_out_aux = nn.Linear(mlp_in, num_classes)
        else:
            #nn.Sequential(nn.Linear(mlp_in, 1024), self.act_end, nn.Linear(1024,1024), self.act_end, nn.Linear(1024, num_classes))
            #nn.Sequential(nn.Linear(mlp_in, 1024), self.act_end, nn.Linear(1024,1024), self.act_end, nn.Linear(1024, num_classes))
            self.mlp_out = FFN(act=self.act_end, norm=args.mlp_norm, nin=mlp_in, nhid=1024, nout=num_classes, mlp_type=args.mlp_type, nlayers=args.mlp_nlayers, act_last=args.mlp_act_last, emb_dim=emb_dim if args.mlp_norm_cond else None, spectral=args.ada_snorm)
            self.mlp_out_aux = FFN(act=self.act_end, norm=args.mlp_norm, nin=mlp_in, nhid=1024, nout=num_classes, mlp_type=args.mlp_type, nlayers=args.mlp_nlayers, act_last=args.mlp_act_last, emb_dim=emb_dim if args.mlp_norm_cond else None, spectral=args.ada_snorm)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.act, self.norm, self.in_planes, planes, stride, dropout = self.dropout, emb_dim=self.emb_dim if self.args.cond_type != 'concatenate' else None, spectral=self.args.ada_snorm))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, t=None, t_img=None, y=None, z=None, std=None, print_=False, lam=None):
        #import pdb; pdb.set_trace()
        if self.per_img_std:
            x = per_image_standardization(x)


        if self.args.add_noise_to_image and t is not None:
            z_input = torch.randn_like(x)
            if self.args.add_noise_to_image_sep:
                mean_input, std_input = self.sde_img.marginal_prob(x, t_img)
            else:
                mean_input, std_input = self.sde_img.marginal_prob(x, t)

            #print('input min max', x.min(), x.max())
            #print('noise std', std_input)

            #print('t', t)
            #print('noise std', std_input)

            #save_image(x, 'before_noise.png', normalize=True)

            x = mean_input + std_input[:, None, None, None] * z_input # b, c, h, w
        
            #x = x.clamp(-2.5, 2.5)
            #save_image(x.clamp(-2.2,2.2), 'after_noise.png', normalize=True)

            #raise Exception('done')

        emb0 = None
        emb1 = None
        if print_:
            print(t)
        if t is not None:
            temb0 = self.TimeNet_0(t, print_=print_)
            if self.args.add_noise_to_image_sep:
                temb0_img = self.TimeNet_0_img(t_img, print_=print_)
            if self.args.mixup_cond:
                temb0_lam = self.TimeNet_0_lam(lam*2, print_=print_) # *2 so in [0,1]
            if self.args.cond_type == 'concatenate':
                temb1 = self.TimeNet_1(t)
                if self.args.add_noise_to_image_sep:
                    temb1_img = self.TimeNet_1_img(t_img, print_=print_)
                if self.args.mixup_cond:
                    temb1_lam = self.TimeNet_1_lam(lam*2, print_=print_) # *2 so in [0,1]
            if y is not None:
                if self.args.yemb_type_special == 0:
                    yemb0 = self.LabelNet_0(y, print_=print_)
                elif self.args.yemb_type_special == 1: # linear(y + noise)
                    yemb0 = self.LabelNet_0(y)
                elif self.args.yemb_type_special == 2: # normalized(linear(y + noise))
                    yemb0 = self.LabelNet_0(y)
                    yemb0 = yemb0 * torch.rsqrt(torch.mean(yemb0 ** 2, dim=1, keepdim=True) + 1e-8)

                if self.args.add_noise_to_image_sep and not self.args.mixup_cond:
                    emb0 = torch.cat([temb0, temb0_img, yemb0], dim=1) # Combine embeddings
                elif self.args.mixup_cond and not self.args.add_noise_to_image_sep:
                    emb0 = torch.cat([temb0, yemb0, temb0_lam], dim=1) # Combine embeddings
                elif self.args.mixup_cond and self.args.add_noise_to_image_sep:
                    emb0 = torch.cat([temb0, temb0_img, yemb0, temb0_lam], dim=1) # Combine embeddings
                else:
                    emb0 = torch.cat([temb0, yemb0], dim=1) # Combine embeddings
                if self.args.cond_type == 'concatenate':
                    yemb1 = self.LabelNet_1(y)
                
                    if self.args.add_noise_to_image_sep and not self.args.mixup_cond:
                        emb1 = torch.cat([temb1, temb1_img, yemb1], dim=1) # Combine embeddings
                    elif self.args.mixup_cond and not self.args.add_noise_to_image_sep:
                        emb1 = torch.cat([temb1, yemb1, temb1_lamb], dim=1) # Combine embeddings
                    elif self.args.mixup_cond and self.args.add_noise_to_image_sep:
                        emb1 = torch.cat([temb1, temb1_img, yemb1, temb1_lamb], dim=1) # Combine embeddings
                    else:
                        emb1 = torch.cat([temb1, yemb1], dim=1) # Combine embeddings
            else:
                emb0 = temb0
                if self.args.cond_type == 'concatenate':
                    emb1 = temb1
            
            if self.args.pred_latent:

                if z is None:
                    z_noise = torch.randn((y.shape[0], 128)).cuda()
                    z_mean, z_std = self.sde.marginal_prob(z_noise*0.0, t)
                    z = z_noise * z_std[:,None]

                zemb0 = self.LatentNet_0(z, print_=print_)
                emb0 = torch.cat([emb0, zemb0], dim=1)

        if self.training and random.random() < self.args.dropout_class: # We randomly dropout the embedding
            emb0 = None
            emb1 = None

        if mixup_hidden:
            layer_mix = random.randint(0,2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None   
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)
        
        if target is not None :
            target_reweighted = target
        
        out = x
        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.conv1(out)
        if emb1 is not None and self.args.cond_type == 'concatenate':
            out = out + self.c1(torch.cat([out, emb1[:,:,None,None].repeat(1,1,out.shape[2],out.shape[3])], dim=1))
        for i, m in enumerate(self.layer1):
            out = m(out, emb0 if self.args.cond_type != 'concatenate' else None)
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        if emb1 is not None and self.args.cond_type == 'concatenate':
            out = out + self.c2(torch.cat([out, emb1[:,:,None,None].repeat(1,1,out.shape[2],out.shape[3])], dim=1))
        for i, m in enumerate(self.layer2):
            out = m(out, emb0 if self.args.cond_type != 'concatenate' else None)
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        if emb1 is not None and self.args.cond_type == 'concatenate':
            out = out + self.c3(torch.cat([out, emb1[:,:,None,None].repeat(1,1,out.shape[2],out.shape[3])], dim=1))
        for i, m in enumerate(self.layer3):
            out = m(out, emb0 if self.args.cond_type != 'concatenate' else None)
        if  layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        if emb1 is not None and self.args.cond_type == 'concatenate':
            out = out + self.c4(torch.cat([out, emb1[:,:,None,None].repeat(1,1,out.shape[2],out.shape[3])], dim=1))
        for i, m in enumerate(self.layer4):
            out = m(out, emb0 if self.args.cond_type != 'concatenate' else None)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        z_aug = self.z_proj_net(out)
        z_aug_pred = self.z_pred_net(z_aug)

        z_pred_ = self.z_pred_net2(out)

        if emb0 is not None: # We must scale it to become a score-function
            assert std is not None

            if emb0 is not None and (self.args.cond_type == 'concatenate' or self.args.cond_type == 'embedding_cat'):
                out_aux = self.mlp_out_aux((torch.cat([out,emb0], dim=1)))
                out = self.mlp_out(torch.cat([out,emb0], dim=1))
            else:
                out_aux = self.mlp_out_aux(out)
                out = self.mlp_out(out)

            out = -out / std[:, None]

            if self.args.multilabel:
                out_normed = F.sigmoid(out_aux)
            else:
                out_normed = F.softmax(out_aux,dim=1)


            if self.args.aux_only:

                out = ((out_normed)-0.5)*2.0
                out = (out - y) / std.unsqueeze(1).repeat(1,y.shape[1])**2
                
        else:
            out_aux = self.mlp_out_aux(out)
            out = self.mlp_out(out)
        
        if self.training:
            return out, out_aux, z_aug, z_aug_pred, z_pred_
        else:
            return out

    def print_weights(self):
        pass
        #if self.args.yemb_type_special == 0:
        #    lst = self.LabelNet_0.all_modules
        #    for m in lst:
        #        print('weight on embedding', m.weight.shape, m.weight.min(), m.weight.max(), m.weight.mean())

def preactresnet9(args, num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(args, PreActBlock, [1,1,1,1], args.initial_channels, num_classes,  per_img_std, stride= stride)

def preactresnet18(args, num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(args, PreActBlock, [2,2,2,2], args.initial_channels, num_classes,  per_img_std, stride= stride)

def preactresnet34(args, num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(args, PreActBlock, [3,4,6,3], args.initial_channels, num_classes,  per_img_std, stride= stride)

def preactresnet50(args, num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(args, PreActBottleneck, [3,4,6,3], 64, num_classes,  per_img_std, stride= stride)

def preactresnet101(args, num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(args, PreActBottleneck, [3,4,23,3], 64, num_classes, per_img_std, stride= stride)

def preactresnet152(args, num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(args, PreActBottleneck, [3,8,36,3], 64, num_classes, per_img_std, stride= stride)

def test():
    net = PreActResNet152(True,10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

if __name__ == "__main__":
    test()
# test()


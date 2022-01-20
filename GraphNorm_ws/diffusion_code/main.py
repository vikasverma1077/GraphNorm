#!/usr/bin/env python
from __future__ import division
import numpy
import os, sys, shutil, time, random
import argparse
from distutils.dir_util import copy_tree
from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import copy
from utils import *
import models
import pdb
import math
import sys
import time
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import numpy as np
from collections import OrderedDict, Counter
from load_data  import *
from helpers import *
from plots import *
from analytical_helper_script import run_test_with_mixup
#from attacks import run_test_adversarial, fgsm, pgd

from sde.sde_lib import VPSDE
from sde.sde_sampling import get_sampling_fn

from torchvision.utils import save_image as save_image

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'k49', 'maze', 'imagenet', 'svhn', 'stl10', 'mnist', 'tiny-imagenet-200', 'shapes'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--data_dir', type = str, default = 'cifar10',
                        help='file where results are to be written')
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')
parser.add_argument('--valid_labels_per_class', type=int, default=0, metavar='NL',
                    help='validation labels_per_class')

parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--initial_channels', type=int, default=64, choices=(8, 16,64))
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--train', type=str, default = 'vanilla', choices =['vanilla','mixup', 'mixup_hidden','cutout'])
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='alpha parameter for mixup')
parser.add_argument('--cutout', type=int, default=16, help='size of cut out')

parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
#parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--valid_batch_size', type=int, default=256, help='Validation Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--do_lr_warmup', type=str2bool, default=True, help='Warmup.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--adv_unpre', action='store_true', default=False,
                     help= 'the adversarial examples will be calculated on real input space (not preprocessed)')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='*', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='*', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--optSteps', type=int, default=1)

# New Score-based stuff
parser.add_argument('--score', action='store_true', default=False, help="Use Score-based models, y(t+h) <- f(x, y(t))")
parser.add_argument('--n_sampling', type=int, default=10, help="Number of times to sample from a single input (100 would be ideal to get good accuracy)")
parser.add_argument('--fourier_scale', type=float, default=16)
parser.add_argument('--score_nf', type=float, default=128)
parser.add_argument('--cond_type', type=str, default='embedding_cat', choices=['embedding','embedding_cat','concatenate','none'], help="Conditioning on noisy y through an embedding or concatenated to the input as an extra channel") # none seems equivalent to embedding based on the if-else of the code
parser.add_argument('--diffusion_num_scales', type=int, default=1000)
parser.add_argument('--diffusion_type', type=str, default='vp', choices=['vp'], help="Only VP will be used.")
parser.add_argument('--vp_beta_min', type=float, default=0.2) #min 0.1
parser.add_argument('--vp_beta_max', type=float, default=20.0) #max 20; 2 is very good

parser.add_argument('--hard_samples', type=str2bool, default=False, help='At end of sampling, makes samples hard one-hots for averaging')

# Adaptive step-size algorithm (Gotta Go Fast)
parser.add_argument('--sampling', type=str, default='adaptive', choices=['euler_maruyama','adaptive'], help="Use the fast adaptive sampler from Gotta Go Fast")
parser.add_argument('--sampling_N', type=int, default=1000, help="Number of sampling steps for Euler-Maruyama")
parser.add_argument('--sampling_h_init', type=float, default=0.015) # was 0.01
parser.add_argument('--sampling_reltol', type=float, default=0.01)
parser.add_argument('--sampling_abstol', type=float, default=0.01) #seems optimal: currently its 2/0.01 = 1/200 difference allowed
parser.add_argument('--sampling_safety', type=float, default=0.8) # was 0.9
parser.add_argument('--sampling_exp', type=float, default=0.9)

parser.add_argument('--bce_loss_weight', type=float, default=1.0)
parser.add_argument('--bce_aux_loss', type=str2bool, default=False)
parser.add_argument('--do_validate', type=str2bool, default=True)
parser.add_argument('--aux_only', type=str2bool, default=False)
parser.add_argument('--embed_norm', type=str2bool, default=False, help="Pixel Normalize the embedding to prevent the embedding being ignored (scale << bias in last layer of embedding cause embedding to be mostly constant)")
parser.add_argument('--embed_snorm', type=str2bool, default=False, help="Spectral-Normalize the embedding to prevent the embedding being ignored (scale << bias in last layer of embedding cause embedding to be mostly constant)")
parser.add_argument('--ada_snorm', type=str2bool, default=False, help="Spectral-Normalize the linear layers of the ada-in to prevent the embedding being ignored and overfitting (scale << bias in last layer of embedding cause embedding to be mostly constant)")

parser.add_argument('--add_noise_to_image', type=str2bool, default=False, help="Also add noise (from the same noise level at the label) to the input image. Will act as a regularization and prevent strong one-shot.")
parser.add_argument('--vp_beta_min_image', type=float, default=0.2) #min 0.1
parser.add_argument('--vp_beta_max_image', type=float, default=20.0) #max 20; 2 is very good
parser.add_argument('--add_noise_to_image_sep', type=str2bool, default=False, help="Add noise to image, but make it a separate embedding for t_image and at testing-time we always use t_image=0 to not add any noise.")

parser.add_argument('--nonlin', type=str, default='swish', choices=['relu','swish','mish'])
parser.add_argument('--norm', type=str, default='batch', choices=['batch','group','evo'])
parser.add_argument('--dropout_p', type=float, default=0.1) # only for preactresnet for now

# MLP out options
parser.add_argument('--mlp_norm', type=str, default='none', choices=['batch','group','layer','none'])
parser.add_argument('--mlp_type', type=str, default='fc', choices=['transformer','resnet','fc'])
parser.add_argument('--mlp_nlayers', type=int, default=1)
parser.add_argument('--mlp_act_last', type=str2bool, default=True)
parser.add_argument('--mlp_norm_cond', type=str2bool, default=False, help="If True the normalization layers will be conditioned on t and y(t)")
parser.add_argument('--mlp_old', type=str2bool, default=False, help="Uses a simple linear layer like before instead of something complex.")

# Regularization tricks
parser.add_argument('--logit_squeeze', type=float, default=0.0, help="")
parser.add_argument('--logit_squeeze_noise', type=float, default=0.0, help="logit squeeze only at t=1")
parser.add_argument('--max_var_emb', type=float, default=0.0, help="max E[(logit_f(x, t, emb) - logit_f(x, t, emb2))^2]")
parser.add_argument('--better_t_small', type=float, default=0.0, help="relativistic: E[loss(logit_f(x, t-h, emb) - logit_f(x, t, emb2))]")
parser.add_argument('--better_t_small2', type=float, default=0.0, help="relativistic: E[loss(logit_f(x, t=0, emb) - logit_f(x, t=1, emb2))] aka t=0 should be bigger in logit-space than t=1 to the correct label")
parser.add_argument('--better_t_small3', type=float, default=0.0, help="relativistic: E[loss(logit_f(x, t=0, emb) - logit_f(x, t=1, emb2))] aka t=0 should be bigger in logit-space than t=1 to the correct label")
parser.add_argument('--better_t_small_hinge', type=float, default=0.0, help="relativistic: E[loss(logit_f(x, t-h, emb) - logit_f(x, t, emb2))]")
parser.add_argument('--better_t_small2_hinge', type=float, default=0.0, help="relativistic: E[loss(logit_f(x, t=0, emb) - logit_f(x, t=1, emb2))] aka t=0 should be bigger in logit-space than t=1 to the correct label")
parser.add_argument('--better_t_small3_hinge', type=float, default=0.0, help="relativistic: E[loss(logit_f(x, t=0, emb) - logit_f(x, t=1, emb2))] aka t=0 should be bigger in logit-space than t=1 to the correct label")
parser.add_argument('--better_t_K', type=float, default=1.0, help="K multiplier for better_t_small2, better_t_small3,  better_t_small2_hinge, and better_t_small3_hinge")

parser.add_argument('--mixup_per_sample', type=str2bool, default=False, help="Mixup is done per sample rather than per batch")
parser.add_argument('--mixup_cond', type=str2bool, default=False, help="Condition classifier on mixup value")

parser.add_argument('--crossentropy', type=str2bool, default=False, help="Use cross-entropy loss which is more stable")

parser.add_argument('--train_eval_freq', type=int, default=200)
parser.add_argument('--val_freq', type=int, default=200)

parser.add_argument('--pred_latent', type=str2bool, default=False, help="Extra loss to predict hidden state with different aug applied")
parser.add_argument('--pred_latent_weight', type=float, default=1.0, help="Extra loss to predict hidden state with different aug applied")
parser.add_argument('--pred_latent_aug', type=str, default='mixup', choices=['mixup','cutout'])
parser.add_argument('--pred_latent_net_type', type=str, default='leaky', choices=['leaky','fc'])


parser.add_argument('--pred_noise', type=str2bool, default=False, help="Extra loss to predict noise")
parser.add_argument('--pred_noise_weight', type=float, default=1.0, help="Extra loss to predict noise")
parser.add_argument('--pred_noise_net_type', type=str, default='leaky', choices=['leaky','fc'])

parser.add_argument('--yemb_type_special', type=int, default=0, help='0 = default, 1 = linear(y+noise), 2 = normalized(linear(y + noise))')

# Classifier-free guidance as per https://openreview.net/pdf?id=qw8AKxfYbI and https://arxiv.org/pdf/2112.10741.pdf
parser.add_argument('--dropout_class', type=float, default=0.0, help='Removes the label embedding with prob = dropout_class during training')
parser.add_argument('--classfree', type=str2bool, default=False, help='Uses classifier-free guidance in sampling')
parser.add_argument('--classfree_w', type=float, default=3) # try 0.1, 1, 2, 3
parser.add_argument('--classfree_base', type=str, default='noclass', choices=['class','noclass'])
# class is original (https://openreview.net/pdf?id=qw8AKxfYbI), noclass is GLIDE (https://arxiv.org/pdf/2112.10741.pdf)


parser.add_argument('--sel_oneshot', type=str2bool, default=False, help='select best model using one-shot accuracy')
# New default below
parser.add_argument('--only_oneshot', type=str2bool, default=True, help='only use and select best model using one-shot accuracy; but still shows multi-shot with the best model')


args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

args.multilabel = (args.dataset == 'maze') or (args.dataset == 'shapes')

out_str = str(args)
print(out_str)

if args.manualSeed is not None:
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = False
else:
    cudnn.benchmark = True

if args.aux_only:
    args.bce_aux_loss = True

if args.add_noise_to_image_sep:
    args.add_noise_to_image = True

if args.better_t_small or args.better_t_small2 or args.better_t_small3:
    args.crossentropy = True

if args.mixup_cond:
    args.mixup_per_sample = True

if (args.mixup_cond or args.mixup_per_sample) and args.train != 'mixup':
    raise Exception('Make train = mixup to use mixup options')



def experiment_name_non_mnist(args,
                    dataset='cifar10',
                    arch='',
                    epochs=400,
                    dropout=True,
                    batch_size=64,
                    lr=0.01,
                    momentum=0.5,
                    decay=0.0005,
                    data_aug=1,
                    train = 'vanilla',
                    mixup_alpha=0.0,
                    job_id=None,
                    add_name=''):
    if add_name != '':
        print('experiment name: ' + add_name)
        return add_name
    exp_name = dataset
    exp_name += '_arch'+str(arch)
    exp_name += '_train'+str(train)
    #exp_name += '_m_alpha'+str(mixup_alpha)
    exp_name += '_epoch'+str(epochs)
    exp_name +='_bs'+str(batch_size)
    #exp_name += '_lr'+str(lr)
    if args.score:
        exp_name += '_score'
        exp_name += '_ns'+str(args.n_sampling)
        exp_name += '_'+str(args.cond_type)
        exp_name += '_vp'+str(args.vp_beta_min)+'_'+str(args.vp_beta_max)
        if args.aux_only:
            exp_name += '_auxonly'
        elif args.bce_aux_loss:
            exp_name += '_aux'
        if args.add_noise_to_image:
            exp_name += '_noiseimg'
        exp_name += '_vpimg'+str(args.vp_beta_min_image)+'_'+str(args.vp_beta_max_image)

    exp_name += '_nonlin'+str(args.nonlin)
    exp_name += '_norm'+str(args.norm)
    exp_name += '_dropp'+str(args.dropout_p)

    exp_name += '_mlp'
    exp_name += str(args.mlp_norm)
    exp_name += '_'+str(args.mlp_type)
    exp_name += '_n'+str(args.mlp_nlayers)
    if args.mlp_act_last:
        exp_name += '_al'
    if args.mlp_norm_cond:
        exp_name += '_nc'
    #exp_name += '_mom_'+str(momentum)
    #exp_name +='_decay_'+str(decay)
    #exp_name += '_data_aug_'+str(data_aug)
    #if job_id!=None:
    #    exp_name += '_job_id_'+str(job_id)
    #if add_name!='':
    #    exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiment name: ' + exp_name)
    return exp_name


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(args, optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break

    if args.do_lr_warmup:
        if epoch < 10:
            lr = 0.0001 + (args.learning_rate * epoch) / 10.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,), args=None):
    """Computes the precision@k for the specified values of k"""


    
    if args.multilabel:

        if args.dataset == 'shapes':
            output_eval = output[:,0:1]
            target_eval = target[:,0:1]
        else:
            output_eval = output
            target_eval = target

        res = []
        for k in topk:
            res.append(torch.eq(target_eval, torch.gt(output_eval,0.5)).float().mean())
        return res

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

bce_loss = nn.BCELoss().cuda()
#hinge_loss = nn.MultiMarginLoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
sigmoid = nn.Sigmoid().cuda()

if args.multilabel:
    out_func = sigmoid
else:
    out_func = softmax

mse_loss = nn.MSELoss().cuda()

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
criterion = SoftTargetCrossEntropy().cuda()
criterion_val = nn.CrossEntropyLoss().cuda() # non-soft for validation

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(target*torch.nn.ReLU()(1.0 - x), dim=-1)
        return torch.mean(loss)
hinge_loss = HingeLoss().cuda()

def train(train_loader, model, ema_model, optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_rec = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    sde = VPSDE(beta_min=args.vp_beta_min, beta_max=args.vp_beta_max, N=args.diffusion_num_scales)
    eps = 1e-5

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        # measure data loading time
        #print (input)
       
        ema_update(model, ema_model, epoch, 0.9999)

        #unique, counts = np.unique(target.numpy(), return_counts=True)
        #print (counts)
        #print(Counter(target.numpy()))
        #if i==100:
        #    break
        #import pdb; pdb.set_trace()
        target = target.long()
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time() - end)
        #import pdb; pdb.set_trace()

        if args.multilabel:
            target_reweighted = target.float()
        else:   
            target_reweighted = F.one_hot(target, args.num_classes).float()

        if args.train == 'mixup':
            if args.mixup_per_sample:
                beta_rand = torch.distributions.beta.Beta(torch.ones(target.shape[0], device=target.device)*args.mixup_alpha, torch.ones(target.shape[0], device=target.device)*args.mixup_alpha)
                lam = beta_rand.sample()
            else:
                lam = get_lambda(args.mixup_alpha)
                lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = torch.maximum(lam, 1-lam)

            lam_aug = get_lambda(args.mixup_alpha)
            lam_aug = torch.from_numpy(np.array([lam_aug]).astype('float32')).cuda()
            lam_aug = torch.maximum(lam_aug, 1-lam_aug)
            mix_ind_aug = torch.randperm(target_reweighted.shape[0])

            mix_ind = torch.randperm(target_reweighted.shape[0])
            target_reweighted = target_reweighted * lam + target_reweighted[mix_ind] * (1-lam)
        else:
            lam = None

        # Get noisy one-hot label
        if args.score:
            target_reweighted_s = (target_reweighted - 0.5) * 2.0 #set range from -1.0 to 1.0
            z = torch.randn_like(target_reweighted_s)
            time_ = torch.rand(target.shape[0], device=target.device) * (sde.T - eps) + eps
            mean, std = sde.marginal_prob(target_reweighted_s, time_)
            perturbed_target = mean + std[:, None] * z # b, c
        else:
            time_ = None
            perturbed_target = None
            std = None
        if args.add_noise_to_image_sep:
            time_img = torch.rand(target.shape[0], device=target.device) * (sde.T - eps) + eps
        else:
            time_img = None

        print_scale_bias = (i == len(train_loader)-1 and epoch % 50 == 0)

        if args.train == 'vanilla' or args.train == 'mixup':
            input_var = Variable(input)

            if args.train == 'mixup':
                if args.mixup_per_sample:
                    input_var_use = lam[:, None, None, None] * input_var + (1-lam[:, None, None, None]) * input_var[mix_ind]
                else:
                    input_var_use = lam * input_var + (1-lam) * input_var[mix_ind]
            else:
                input_var_use = input_var

            if args.score:
                
                if args.pred_latent:
                    if args.pred_latent_aug == "mixup":
                        input_aug = lam_aug * input_var + (1-lam_aug) * input_var[mix_ind_aug]
                    elif args.pred_latent_aug == "cutout":
                        cutout = Cutout(1, args.cutout)
                        input_aug = cutout.apply(input_var)
                    else:
                        raise Exception("pred_latent_aug chosen is not a valid option")
                    _, _, z_aug, _, _ = ema_model(input_aug)

                    z_noise = torch.randn_like(z_aug)
                    z_mean, z_std = sde.marginal_prob(z_aug, time_)
                    z_noise = z_mean + z_std[:,None] * z_noise
                    output, output_aux, _, z_aug_pred, z_pred = model(input_var_use, t=time_, t_img=time_img, y=perturbed_target, z=z_noise.detach(), std=std, print_=print_scale_bias, lam=lam, target=target_reweighted)
                else:
                    if print_scale_bias: # we want to test model over different t and over different y to see if it changes
                        with torch.no_grad():
                            if args.multilabel:
                                dim_ = 2
                            else:
                                dim_ = 1
                            for tt in [0.01, 0.25, 0.50, 0.75, 0.99]:
                                time_test = torch.ones(target.shape[0], device=target.device) * tt
                                mean_test, std_test = sde.marginal_prob(target_reweighted_s, time_test)
                                perturbed_target_test = mean_test + std_test[:, None] * z # b, c
                                output_test, output_aux_test, _, _, _ = model(input_var_use, t=time_test, t_img=time_img, y=perturbed_target_test, std=std, print_=False, lam=lam)
                                if not args.aux_only:
                                    output_test = output_aux_test
                                if tt == 0.01:
                                    if args.multilabel:
                                        output_test_cat = output_test[...,None]
                                    else:
                                        output_test_cat = output_test[range(output_test.shape[0]),target][:,None]
                                else:
                                    if args.multilabel:
                                        output_test_cat = torch.cat([output_test_cat, output_test[...,None]], dim=dim_)
                                    else:
                                        output_test_cat = torch.cat([output_test_cat, output_test[range(output_test.shape[0]),target][:,None]], dim=dim_)
                            print(f'Var(y[:,target], dim=[t=0.01, 0.25, 0.50, 0.75, 1]): {torch.var(output_test_cat, unbiased=False, dim=dim_)}')
                            for oo in [0.01, 0.25, 0.50, 0.75, 0.99]:
                                z_test = torch.randn_like(target_reweighted_s)
                                perturbed_target_test = mean + std[:, None] * z_test # b, c
                                output_test, output_aux_test, _, _, _ = model(input_var_use, t=time_, t_img=time_img, y=perturbed_target_test, std=std, print_=False, lam=lam)
                                if not args.aux_only:
                                    output_test = output_aux_test
                                if oo == 0.01:
                                    if args.multilabel:
                                        output_test_cat = output_test[...,None]
                                    else:
                                        output_test_cat = output_test[range(output_test.shape[0]),target][:,None]
                                else:
                                    if args.multilabel:
                                        output_test_cat = torch.cat([output_test_cat, output_test[...,None]], dim=dim_)
                                    else:
                                        output_test_cat = torch.cat([output_test_cat, output_test[range(output_test.shape[0]),target][:,None]], dim=dim_)
                            print(f'Var(y[:,target], dim=[5 random z with same t]): {torch.var(output_test_cat, unbiased=False, dim=dim_)}')

                    output, output_aux, _, z_aug_pred, z_pred = model(input_var_use, t=time_, t_img=time_img, y=perturbed_target, std=std, print_=print_scale_bias, lam=lam)

                    #rounds = np.random.randint(args.optSteps) + 1
                    #for _ in range(rounds):
                    #    output, output_aux, _, z_aug_pred, z_pred = model(input_var_use, t=time_, t_img=time_img, y=perturbed_target, std=std, print_=print_scale_bias, lam=lam)
                    #    target_reweighted_s = (F.softmax(output, dim=1) - 0.5) * 2.0 #set range from -1.0 to 1.0
                    #    z = torch.randn_like(target_reweighted_s)
                    #    time_ = time_ * 0.9
                    #    mean, std = sde.marginal_prob(target_reweighted_s, time_)
                    #    perturbed_target = mean + std[:, None] * z # b, c
                        
            else:
                output, output_aux, _, _, _ = model(input_var_use)
            #loss = criterion(output, target_var)
            #target_one_hot = to_one_hot(target_var, args.num_classes)
            if not args.score:
                if args.crossentropy:
                    loss = criterion(output, target_reweighted)
                else:
                    loss = bce_loss(out_func(output), target_reweighted)
            else:
                if not args.aux_only:
                    # output is in R^n_classes and continuous, we want it equal it equal to -z after multiplication by std
                    losses = torch.square(output * std[:, None] + z)
                    losses = torch.mean(losses, dim=(1)) # mean over channels (n_classes)
                    loss = 0.5 * torch.mean(losses) # Mean over batch-size
                    output = perturbed_target + output * std[:, None]**2 # Expected Denoised Label
                else:
                    loss = 0.0
                    output = output_aux # accuracy will be based on aux

                if args.pred_latent:
                    loss += mse_loss(F.normalize(z_aug_pred), F.normalize(z_aug.detach())) * args.pred_latent_weight

                if args.score and args.pred_noise:
                    loss += mse_loss(z, z_pred) * args.pred_noise_weight

                if args.bce_aux_loss:
                    if args.crossentropy:
                        loss += args.bce_loss_weight * criterion(output_aux, target_reweighted)
                    else:
                        loss += args.bce_loss_weight * bce_loss(out_func(output_aux), target_reweighted)
                
                    # Regularizations
                    perturbed_target_1 = None
                    if args.logit_squeeze != 0:
                        loss += args.logit_squeeze * torch.mean(output_aux ** 2)
                    if args.logit_squeeze_noise != 0:
                        time_1 = torch.rand(target.shape[0], device=target.device) * sde.T
                        mean_1, std_1 = sde.marginal_prob(target_reweighted_s, time_1)
                        perturbed_target_1 = mean_1 + std_1[:, None] * z # b, c
                        _, output_aux_t1, _, _, _ = model(input_var_use, t=time_1, t_img=time_img, y=perturbed_target_1, std=std_1, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.logit_squeeze_noise * torch.mean(output_aux_t1 ** 2)
                    if args.max_var_emb != 0:
                        z2 = torch.randn_like(target_reweighted_s)
                        perturbed_target2 = mean + std[:, None] * z2 # b, c
                        _, output_aux2, _, _, _ = model(input_var_use, t=time_, t_img=time_img, y=perturbed_target2, std=std, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss -= args.max_var_emb * torch.mean((output_aux - output_aux2) ** 2)
                    if args.better_t_small != 0:
                        time_h = torch.rand(target.shape[0], device=target.device) * (time_ - eps) + eps
                        mean_more, std_more = sde.marginal_prob(target_reweighted_s, time_ + time_h)
                        perturbed_target_more = mean_more + std_more[:, None] * z # b, c
                        _, output_aux_more, _, _, _ = model(input_var_use, t=time_+time_h, t_img=time_img, y=perturbed_target_more, std=std_more, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.better_t_small * criterion(output_aux - output_aux_more, target_reweighted) # t small must be more realistic
                    if args.better_t_small2 != 0:
                        if perturbed_target_1 is None:
                            time_1 = torch.rand(target.shape[0], device=target.device) * sde.T
                            mean_1, std_1 = sde.marginal_prob(target_reweighted_s, time_1)
                            perturbed_target_1 = mean_1 + std_1[:, None] * z # b, c
                        time_0 = torch.rand(target.shape[0], device=target.device) * eps
                        mean_0, std_0 = sde.marginal_prob(target_reweighted_s, time_0)
                        perturbed_target_0 = mean_0 + std_0[:, None] * z # b, c
                        _, output_aux_0, _, _, _ = model(input_var_use, t=time_0, t_img=time_img, y=perturbed_target_0, std=std_0, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        _, output_aux_1, _, _, _ = model(input_var_use, t=time_1, t_img=time_img, y=perturbed_target_1, std=std_1, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.better_t_small2 * criterion(args.better_t_K*(output_aux_0 - output_aux_1), target_reweighted) # t=0 must be more realistic than t=1
                    if args.better_t_small3 != 0:
                        time_h = torch.rand(target.shape[0], device=target.device) * (time_ - eps) + eps
                        mean_more, std_more = sde.marginal_prob(target_reweighted_s, time_ + time_h)
                        perturbed_target_more = mean_more + std_more[:, None] * z # b, c
                        _, output_aux_more, _, _, _ = model(input_var_use, t=time_+time_h, t_img=time_img, y=perturbed_target_more, std=std_more, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.better_t_small3 * criterion(time_h*args.better_t_K*(output_aux - output_aux_more), target_reweighted) # t small must be more realistic
                    if args.better_t_small_hinge != 0:
                        time_h = torch.rand(target.shape[0], device=target.device) * (time_ - eps) + eps
                        mean_more, std_more = sde.marginal_prob(target_reweighted_s, time_ + time_h)
                        perturbed_target_more = mean_more + std_more[:, None] * z # b, c
                        _, output_aux_more, _, _, _ = model(input_var_use, t=time_+time_h, t_img=time_img, y=perturbed_target_more, std=std_more, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.better_t_small_hinge * hinge_loss(output_aux - output_aux_more, target_reweighted)
                    if args.better_t_small2_hinge != 0:
                        if perturbed_target_1 is None:
                            time_1 = torch.rand(target.shape[0], device=target.device) * sde.T
                            mean_1, std_1 = sde.marginal_prob(target_reweighted_s, time_1)
                            perturbed_target_1 = mean_1 + std_1[:, None] * z # b, c
                        time_0 = torch.rand(target.shape[0], device=target.device) * eps
                        mean_0, std_0 = sde.marginal_prob(target_reweighted_s, time_0)
                        perturbed_target_0 = mean_0 + std_0[:, None] * z # b, c
                        _, output_aux_0, _, _, _ = model(input_var_use, t=time_0, t_img=time_img, y=perturbed_target_0, std=std_0, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        _, output_aux_1, _, _, _ = model(input_var_use, t=time_1, t_img=time_img, y=perturbed_target_1, std=std_1, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.better_t_small2_hinge * hinge_loss(args.better_t_K*(output_aux_0 - output_aux_1), target_reweighted)
                    if args.better_t_small3_hinge != 0:
                        time_h = torch.rand(target.shape[0], device=target.device) * (time_ - eps) + eps
                        mean_more, std_more = sde.marginal_prob(target_reweighted_s, time_ + time_h)
                        perturbed_target_more = mean_more + std_more[:, None] * z # b, c
                        _, output_aux_more, _, _, _ = model(input_var_use, t=time_+time_h, t_img=time_img, y=perturbed_target_more, std=std_more, print_=print_scale_bias, lam=lam, target=target_reweighted)
                        loss += args.better_t_small3_hinge * hinge_loss(time_h*args.better_t_K*(output_aux - output_aux_more), target_reweighted)

        elif args.train == 'cutout':
            cutout = Cutout(1, args.cutout)
            cut_input = cutout.apply(input)
                
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            cut_input_var = torch.autograd.Variable(cut_input)
            #if dataname== 'mnist':
            #    input = input.view(-1, 784)
            if args.score:
                output, output_aux, _, _, _ = model(cut_input_var, t=time_, t_img=time_img, y=perturbed_target, std=std, print_=print_scale_bias, target=target_reweighted)
            else:
                output, output_aux, _, _, _ = model(cut_input_var)
            #loss = criterion(output, target_var)
        
            if not args.score:
                if args.crossentropy:
                    loss = criterion(output, target_reweighted)
                else:
                    loss = bce_loss(softmax(output), target_reweighted)
            else:
                if not args.aux_only:
                    # output is in R^n_classes and continuous, we want it equal it equal to -z after multiplication by std
                    losses = torch.square(output * std[:, None] + z)
                    losses = torch.mean(losses, dim=(1)) # mean over channels (n_classes)
                    loss = 0.5 * torch.mean(losses) # Mean over batch-size
                    output = perturbed_target + output * std[:, None]**2 # Expected Denoised Label
                else:
                    loss = 0.0
                    output = output_aux # accuracy will be based on aux

                if args.bce_aux_loss:
                    if args.crossentropy:
                        loss += args.bce_loss_weight * criterion(output_aux, target_reweighted)
                    else:
                        loss += args.bce_loss_weight * bce_loss(softmax(output_aux), target_reweighted)

        sys.stdout.flush()
        # measure accuracy and record loss
        losses_rec.update(loss.item(), input.size(0))
        prec1, prec5 = accuracy(output, target, topk=(1, 5), args=args)
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses_rec, top1=top1, top5=top5) + time_string(), log)
            ## Testing: print the EDS of the first sample, should get closer to -1 -1 -1 1 -1 over time unless it cannot be solved in one-shot
            #n_digits = 3
            #print(torch.round(output[0][:].data * 10**n_digits) / (10**n_digits))
          
        
              
        if args.score and i == len(train_loader)-1 and epoch > 0 and epoch % args.train_eval_freq == 0:

            #save_image(input_var, 'train_img.png')

            model.eval()
            sampling_fn = get_sampling_fn(opt = args, sde = sde, shape = (target.size(0), args.num_classes), eps = 1e-3, device = output.device, N=args.sampling_N)
            sampling_fn_oneshot = get_sampling_fn(opt = args, sde = sde, shape = (target.size(0), args.num_classes), eps = 1e-3, device = output.device, N=0)

            ## Testing: print the EDS of the first sample, should get closer to -1 -1 -1 1 -1 over time unless it cannot be solved in one-shot

            output, n_eval = 0, 0
            output_oneshot, n_eval_oneshot = 0, 0
            for _ in range(args.n_sampling):
                with torch.no_grad():
                    output_i, n_eval_i = sampling_fn(model, ema_model, data=input_var)
                if args.hard_samples:
                    #output += F.softmax(20.0 * ((output_i / 2) + 0.5), dim=1)
                    hard_sample = torch.argmax(output_i, dim=1)
                    output += F.one_hot(hard_sample, args.num_classes).float()
                else:                
                    output += torch.clamp((output_i / 2) + 0.5, 0.0, 1.0)

                n_eval += n_eval_i

                output_i_oneshot, n_eval_i_oneshot = sampling_fn_oneshot(model, ema_model, data=input_var)
                output_oneshot += (output_i_oneshot / 2) + 0.5
                n_eval_oneshot += n_eval_i_oneshot
            output /= args.n_sampling
            n_eval /= args.n_sampling
            output_oneshot /= args.n_sampling
            n_eval_oneshot /= args.n_sampling

            print('output0', output[0])
            #print('output0-oneshot', output_oneshot[0])

            print('output min/mean/max', output[0].min(), output[0].mean(), output[0].max())

            print('acc all zeros', accuracy(output*0.0, target, topk = (1,5), args=args))

            n_digits = 3
            print((torch.round(output[0][:].data * 10**n_digits) / (10**n_digits)).cpu().numpy())

            prec1, prec5 = accuracy(output, target, topk=(1, 5), args=args)
            prec1_oneshot, prec5_oneshot = accuracy(output_oneshot, target, topk=(1, 5), args=args)

            #model.print_weights()

            print_log(' <Sampling> Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1:.3f}  '
                'Prec@5 {top5:.3f}  '
                'Prec@1 (one-shot) {top1_oneshot:.3f}  '
                'Prec@5 (one-shot) {top5_oneshot:.3f}  '
                'Sampling NEF {nef}   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses_rec, top1=prec1.item(), top5=prec5.item(), top1_oneshot=prec1_oneshot.item(), top5_oneshot=prec5_oneshot.item(), nef=n_eval) + time_string(), log)
            model.train()

    #print_log(' **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, top5.avg, losses_rec.avg


def validate(val_loader, args, model, ema_model, log, full=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_oneshot = AverageMeter()
    top5_oneshot = AverageMeter()

    valid_time = time.time()
    # switch to evaluate mode
    model.eval()

    bc = 0

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
          target = target.cuda()
          input = input.cuda()
        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

        if not args.score:
            output = model(input_var)
            if not args.multilabel:
                loss = criterion_val(output, target_var)
            else:
                loss = torch.Tensor([0.0])
        else:
            sde = VPSDE(beta_min=args.vp_beta_min, beta_max=args.vp_beta_max, N=args.diffusion_num_scales)
            sampling_fn = get_sampling_fn(opt = args, sde = sde, shape = (target.size(0), args.num_classes), eps = 1e-3, device = input_var.device, N=args.sampling_N)
            sampling_fn_oneshot = get_sampling_fn(opt = args, sde = sde, shape = (target.size(0), args.num_classes), eps = 1e-3, device = input_var.device, N=0)

            #save_image(input_var, 'test_image.png')
            #raise Exception('done')

            output, n_eval = 0, 0
            output_oneshot, n_eval_oneshot = 0, 0
            # Average over multiple sampling
            for _ in range(args.n_sampling):

                if (not args.only_oneshot) or full: # multi-shot
                    output_i, n_eval_i = sampling_fn(model, ema_model, data=input_var)
                    #output += F.softmax(10.0 * ((output_i / 2) + 0.5), dim=1)
                    #output += torch.clamp(((output_i / 2) + 0.5), 0.0, 1.0)
                    if args.hard_samples:
                        #output += F.softmax(20.0 * ((output_i / 2) + 0.5), dim=1)
                        hard_sample = torch.argmax(output_i, dim=1)
                        output += F.one_hot(hard_sample, args.num_classes).float()
                    else:
                        #output += F.softmax(10.0 * ((output_i / 2) + 0.5), dim=1)
                        output += torch.clamp((output_i / 2) + 0.5, 0.0, 1.0)
                    n_eval += n_eval_i

                output_i_oneshot, n_eval_i_oneshot = sampling_fn_oneshot(model, ema_model, data=input_var)
                if args.hard_samples:
                    hard_sample_oneshot = torch.argmax(output_i_oneshot, dim=1)
                    output_oneshot += F.one_hot(hard_sample_oneshot, args.num_classes).float()
                else:
                    #output_oneshot += (output_i_oneshot / 2) + 0.5
                    output_oneshot += torch.clamp((output_i_oneshot / 2) + 0.5, 0.0, 1.0)
                
                n_eval_oneshot += n_eval_i_oneshot
            output_oneshot /= args.n_sampling
            n_eval_oneshot /= args.n_sampling
            if (not args.only_oneshot) or full:
                output /= args.n_sampling
                n_eval /= args.n_sampling
            else: # We don't multi-shot
                output = output_oneshot
                n_eval = n_eval_oneshot

            if not args.multilabel:
                loss = criterion_val(output, target_var)
            else:
                loss = torch.Tensor([0.0])

            prec1_oneshot, prec5_oneshot = accuracy(output_oneshot, target, topk=(1, 5), args=args)
            top1_oneshot.update(prec1_oneshot.item(), input.size(0))
            top5_oneshot.update(prec5_oneshot.item(), input.size(0))

        losses.update(loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, topk=(1, 5), args=args)
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        bc += 1
        if (not full) and bc >= 4 and (not args.only_oneshot):
            break

    #print('output0', output[0])
    #print('output0-oneshot', output_oneshot[0])
    #print('output min/mean/max', output[0].min(), output[0].mean(), output[0].max())
    #print('acc all zeros', accuracy(output*0.0, target, topk = (1,5), args=args))

    print("Full validation?", full)
    print("Time to validate", time.time() - valid_time)

    if not args.score:
        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)
    else:
        ## Testing: print the EDS of the first sample, should get closer to -1 -1 -1 1 -1 over time unless it cannot be solved in one-shot
        n_digits = 3
        print(torch.round(output[0][:].data * 10**n_digits) / (10**n_digits))
        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Prec@1 (one-shot) {top1_oneshot.avg:.3f} Prec@5 (one-shot) {top5_oneshot.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} Sampling NEF {nef}'.format(top1=top1, top5=top5, top1_oneshot=top1_oneshot, top5_oneshot=top5_oneshot, error1=100-top1.avg, losses=losses, nef=n_eval), log)
    model.train()

    if args.sel_oneshot and args.score:
        return top1_oneshot.avg, losses.avg
    else:
        return top1.avg, losses.avg

best_acc = 0
def main():

    ### set up the experiment directories########
    exp_name=experiment_name_non_mnist(args,
                    dataset=args.dataset,
                    arch=args.arch,
                    epochs=args.epochs,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    decay= args.decay,
                    data_aug=args.data_aug,
                    train = args.train,
                    mixup_alpha = args.mixup_alpha,
                    job_id=args.job_id,
                    add_name=args.add_name)
    
    exp_dir = args.root_dir+exp_name

    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    
    copy_script_to_folder(os.path.abspath(__file__), exp_dir)

    result_png_path = os.path.join(exp_dir, 'results.png')


    global best_acc

    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(exp_dir), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    
    if args.adv_unpre:
        per_img_std = True
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset_unpre(args.data_aug, args.batch_size, args.valid_batch_size, args.workers ,args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class)
    else:
        per_img_std = False
        train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, args.valid_batch_size, args.workers, args.dataset, args.data_dir,  labels_per_class = args.labels_per_class, valid_labels_per_class = args.valid_labels_per_class)
    args.num_classes = num_classes

    if args.dataset == 'tiny-imagenet-200':
        stride = 2 
    else:
        stride = 1
    #train_loader, valid_loader, _ , test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size, 2, args.dataset, args.data_dir, 0.0, labels_per_class=5000)
    print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](args, num_classes, args.dropout, per_img_std).cuda()
    print_log("=> network :\n {}".format(net), log)

    #net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    ema_net = copy.deepcopy(net).cuda()
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

    #optimizer = torch.optim.AdamW(net.parameters(), state['learning_rate'], weight_decay=state['decay'])

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        with torch.no_grad():
            for test_name in test_loader:
                print(test_name)
                validate(test_loader[test_name], args, net, ema_net, log, full=True)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    # Main loop
    train_loss = []
    train_acc=[]
    test_loss=[]
    test_acc=[]
    
    val_acc, val_los = 0.5, 0.5
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        tr_acc, tr_acc5, tr_los  = train(train_loader, net, ema_net, optimizer, epoch, args, log)

        is_best = False

        # evaluate on validation set
        if args.do_validate and epoch != 0 and (epoch % args.val_freq == 0 or epoch == args.epochs - 1):
            with torch.no_grad():
                for test_name in test_loader:
                    print(test_name)
                    val_acc, val_los = validate(test_loader[test_name], args, net, ema_net, log)
            test_loss.append(val_los)
            test_acc.append(val_acc)
            dummy = recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)

            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc

            save_checkpoint({
              'epoch': epoch + 1,
              'arch': args.arch,
              'state_dict': net.state_dict(),
              'recorder': recorder,
              'optimizer' : optimizer.state_dict(),
            }, is_best, exp_dir, 'checkpoint.pth.tar')

        train_loss.append(tr_los)
        train_acc.append(tr_acc)


        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve(result_png_path)
    
        #import pdb; pdb.set_trace()
        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc']=train_acc
        train_log['test_loss']=test_loss
        train_log['test_acc']=test_acc
        
                   
        pickle.dump(train_log, open( os.path.join(exp_dir,'log.pkl'), 'wb'))
        #plotting(exp_dir)

    if args.do_validate and args.epochs != 1:
        print("Validation from the best model")
        checkpoint = torch.load(os.path.join(exp_dir, 'model_best.pth.tar'))
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        with torch.no_grad():
            for test_name in test_loader:
                print(test_name)
                validate(test_loader[test_name], args, net, ema_net, log, full=True)

    log.close()


if __name__ == '__main__':
    main()

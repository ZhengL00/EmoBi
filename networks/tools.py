import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm, eps=1e-6, elementwise_affine=True)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
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
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def diagnose_network(net, name='network'):

    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


class MidLayerFeatureExtractor(object):
    def __init__(self, layer):
        self.layer = layer
        self.feature = None
        self.layer.register_forward_hook(self.hook)
        self.device = None

    def hook(self, module, input, output):
        self.is_empty = True
        self.feature = output.clone()
        self.is_empty = False

    def extract(self):
        assert not self.is_empty, 'Synic Error in MidLayerFeatureExtractor, \
                this may caused by calling extract method before the hooked module has execute forward method'
        return self.feature


class MultiLayerFeatureExtractor(object):
    def __init__(self, net, layers):

        self.net = net
        self.layer_names = layers.strip().split(',')
        self.layers = [self.str2layer(layer_name) for layer_name in self.layer_names]
        self.extractors = [MidLayerFeatureExtractor(layer) for layer in self.layers]

    def str2layer(self, name):
        modules = name.split('.')
        layer = self.net
        for module in modules:
            if '[' and ']' in module:
                sequential_name = module[:module.find('[')]
                target_module_num = int(module[module.find('[') + 1:module.find(']')])
                layer = getattr(layer, sequential_name)
                layer = layer[target_module_num]
            else:
                layer = getattr(layer, module)

        return layer

    def extract(self):
        ans = [extractor.extract() for extractor in self.extractors]
        return ans


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

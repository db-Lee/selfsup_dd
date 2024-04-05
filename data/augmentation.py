import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia.augmentation as K

NUM_CLASSES = {
    'svhn': 10,
    'cifar10': 10,
    'cifar100': 100,
    'aircraft': 100,
    'cars': 196,
    'cub2011': 200,
    'dogs': 120,
    'flowers': 102,
    'tinyimagenet': 200,
    'imagenet': 1000,
    'imagenette': 10
}

MEAN = {
    32:
    {'svhn': (0.4377, 0.4438, 0.4728),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4866, 0.4409),
    'aircraft': (0.4804, 0.5116, 0.5349),
    'cars': (0.4706, 0.4600, 0.4548),
    'cub2011': (0.4857, 0.4995, 0.4324),
    'dogs': (0.4765, 0.4516, 0.3911),
    'flowers': (0.4344, 0.3030, 0.2955),
    'tinyimagenet': (0.4802, 0.4481, 0.3975),
    'imagenet': (0.4810, 0.4574, 0.4078)},
    64:
    {'svhn': (0.4377, 0.4438, 0.4728),
    'cifar10': (0.4914, 0.4821, 0.4465),
    'cifar100': (0.5070, 0.4865, 0.4409),
    'aircraft': (0.4797, 0.5108, 0.5341),
    'cars': (0.4707, 0.4601, 0.4549),
    'cub2011': (0.4856, 0.4995, 0.4324),
    'dogs': (0.4765, 0.4517, 0.3911),
    'flowers': (0.4344, 0.3830, 0.2955),
    'tinyimagenet': (0.4802, 0.4481, 0.3975),
    'imagenet': (0.4810, 0.4574, 0.4078)},
    224:
    {'aircraft': (0.4797, 0.5109, 0.5342),
    'cars': (0.4707, 0.4601, 0.4549),
    'cub2011': (0.4856, 0.4994, 0.4324),
    'dogs': (0.4765, 0.4517, 0.3912),
    'flowers': (0.4344, 0.3830, 0.2955),
    'imagenette': (0.4655, 0.4546, 0.4250)}
}
STD = {
    32:
    {'svhn': (0.1980, 0.2010, 0.1970),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'aircraft': (0.2021, 0.1953, 0.2297),
    'cars': (0.2746, 0.2740, 0.2831),
    'cub2011': (0.2145, 0.2098, 0.2496),
    'dogs': (0.2490, 0.2435, 0.2479), 
    'flowers': (0.2811, 0.2318, 0.2607),   
    'tinyimagenet': (0.2770, 0.2691, 0.2821),
    'imagenet': (0.2633, 0.2560, 0.2708)},
    64:
    {'svhn': (0.1981, 0.2011, 0.1971),
    'cifar10': (0.2469, 0.2433, 0.2614),
    'cifar100': (0.2671, 0.2562, 0.2760),
    'aircraft': (0.2117, 0.2049, 0.2380),
    'cars': (0.2836, 0.2826, 0.2914),
    'cub2011': (0.2218, 0.2170, 0.2564),
    'dogs': (0.2551, 0.2495, 0.2539),
    'flowers': (0.2878, 0.2390, 0.2674),
    'tinyimagenet': (0.2770, 0.2691, 0.2821),
    'imagenet': (0.2633, 0.2560, 0.2708)},
    224:
    {'aircraft': (0.2204, 0.2135, 0.2451),
    'cars': (0.2927, 0.2917, 0.3001),
    'cub2011': (0.2295, 0.2250, 0.2635),
    'dogs': (0.2617, 0.2564, 0.2607),
    'flowers': (0.2928, 0.2449, 0.2726),
    'imagenette': (0.2804, 0.2754, 0.2965)}
}

def get_aug(data_name, size, aug=True):
    if aug:
        if data_name == "svhn":
            transform_tr = nn.Sequential(
                K.RandomCrop(size=(size,size), padding=4),
                K.Normalize(MEAN[size][data_name], STD[size][data_name])
            )
        else:   
            transform_tr = nn.Sequential(                
                K.RandomCrop(size=(size,size), padding=4),
                K.RandomHorizontalFlip(p=0.5),
                K.Normalize(MEAN[size][data_name], STD[size][data_name]),                
            )        
    else:
        transform_tr = K.Normalize(MEAN[size][data_name], STD[size][data_name])
    
    transform_te = K.Normalize(MEAN[size][data_name], STD[size][data_name])

    return transform_tr, transform_te

"""DC Augmentation"""
class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone().detach()
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone().detach()
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone().detach()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone().detach()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone().detach()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone().detach()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone().detach()
        translation_y[:] = translation_y[0].clone().detach()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone().detach()
        offset_y[:] = offset_y[0].clone().detach()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij"
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}
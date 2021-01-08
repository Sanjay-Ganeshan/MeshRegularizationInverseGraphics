import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import math
import numpy as np

import pyredner as pyr

from .. import pxtyping as T

def bhwc_to_bchw(images):
    bchw = torch.transpose(images, 3, 1)
    # Now [0, 3, 2, 1]
    bchw = torch.transpose(bchw, 3, 2)
    # Now [0, 3, 1, 2]
    return bchw

def bchw_to_bhwc(images):
    # Now [0, 3, 1, 2]
    bhwc = torch.transpose(images, 3, 2)
    # Now [0, 3, 2, 1]
    bhwc = torch.transpose(bhwc, 3, 1)
    # Now [0, 1, 2, 3]
    return bhwc

class GaussianBlur(nn.Module):
    def __init__(self, sigma = 1, kernel_size = 5, channels = 4):
        super().__init__()
        self.filter = make_gaussian_filter(sigma, kernel_size, channels)
    def forward(self, images):
        # Convert from BHWC to BCHW
        # We want to go from order [0,1,2,3] to [0,3,1,2]
        bchw = bhwc_to_bchw(images)
        # Conv' it
        blurry = self.filter(bchw)
        # Want to go back
        # Current [0,3,1,2]
        blurry_bhwc = bchw_to_bhwc(blurry)
        return blurry_bhwc

def make_gaussian_filter(sigma, kernel_size, channels):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding = kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

class Resize(nn.Module):
    '''
    Resizes an image arbitrarily
    '''
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, image):
        bchw = bhwc_to_bchw(image)
        res = F.interpolate(bchw, scale_factor=self.scale, mode='bilinear', recompute_scale_factor=False, align_corners=False)
        res_bhwc = bchw_to_bhwc(res)
        return res_bhwc

class HalfSize(Resize):
    '''
    Halves the size of the image in both dimensions
    '''
    def __init__(self):
        super().__init__(0.5)

class DoubleSize(Resize):
    '''
    Doubles the size of the image in both dimensions
    '''
    def __init__(self):
        super().__init__(2.0)

def compute_average_color(bhwc_rgba_images):
    # Average using only the non-transparent pixels
    # We're okay averaging across BHW, not C, though
    rgb = bhwc_rgba_images[:,:,:,:3]
    a = bhwc_rgba_images[:,:,:,3]
    weighted_rgb = rgb * a.unsqueeze(3)
    total_a = torch.sum(a)
    # Don't sum R,G and B
    total_rgb = torch.sum(weighted_rgb.view((-1, 3)), dim=0)    
    # Now we have: Total R, total G, total B, total A. We 
    # assume when a=0, RGB = 0
    avg_rgb = total_rgb / total_a
    return avg_rgb

def noise_texture(resolution = 1024) -> T.Texture:
    texels = torch.empty((resolution, resolution, 3), device=pyr.get_device(), dtype=torch.float)
    torch.nn.init.uniform_(texels)
    tex = pyr.Texture(texels.to(pyr.get_device()))
    return tex

def average_color_texture(bhwc_rgba_images: T.Images, resolution=1024):
    avg_color = compute_average_color(bhwc_rgba_images)
    # avg color is shape (3,)
    texels = torch.ones((resolution, resolution, 3), device=pyr.get_device(), dtype=torch.float)
    texels *= avg_color.view((1, 1, 3))
    tex = pyr.Texture(texels.to(pyr.get_device()))
    return tex

def copy_texture(orig_tex: T.Optional[T.Texture]):
    if orig_tex is None:
        return None
    new_texels = orig_tex.texels.clone().detach()
    return pyr.Texture(new_texels)

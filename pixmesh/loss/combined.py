import torch
import torch.nn as nn
import pyredner as pyr

from ..data import load_3d, load_settings, load_target_images, make_scenes
from ..util.render import render

class CombinedLoss(nn.Module):
    def __init__(self, target_name, target_is_image = False):
        super().__init__()
        self.target_name = target_name
        if target_is_image:
            self.target_rendered = load_target_images(self.target_name)
        else:
            self.target_mesh = load_3d(self.target_name)
            self.target_settings = load_settings(self.target_name)
            self.target_scenes = make_scenes(self.target_mesh, self.target_settings)
            self.target_rendered = render(self.target_scenes, self.target_settings, grad=False, alpha=True)

        self.losses_3d = {}
        self.loss_weights_3d = {}

        self.losses_2d = {}
        self.loss_weights_2d = {}

        self.breakdown = None

    def forward(self, output_3d, output_rendered):
        total = None
        breakdown = {}
        for name_3d in self.losses_3d:
            (scale, fn) = self.losses_3d[name_3d]
            weight = self.loss_weights_3d.get(name_3d, 1.0)
            if weight < 0.0000001:
                continue
            this_loss = fn(output_3d) * scale
            breakdown[name_3d] = float(this_loss.item())
            
            # Add to total
            if total is None:
                total = this_loss * weight
            else:
                total = total + (this_loss * weight)
        
        for name_2d in self.losses_2d:
            (scale, fn) = self.losses_2d[name_2d]
            weight = self.loss_weights_2d.get(name_2d, 1.0)
            if weight < 0.0000001:
                continue
            this_loss = fn(output_rendered, self.target_rendered) * scale
            breakdown[name_2d] = float(this_loss.item())
            
            # Add to total
            if total is None:
                total = this_loss * weight
            else:
                total = total + (this_loss * weight)
        
        
        if total is not None:
            breakdown['_total'] = float(total.item())

        # This doesn't actually have to do with the model,
        # so let's save it as a field
        self.breakdown = breakdown
        return total

    def get_targets(self):
        return self.target_rendered

    def get_last_breakdown(self):
        if self.breakdown is not None:
            return {k:self.breakdown[k] for k in self.breakdown}
        else:
            return None

    def add_3d(self, name, scaling_factor, loss_fn):
        self.losses_3d[name] = (scaling_factor, loss_fn)
        self.loss_weights_3d[name] = 1.0
        return self
    
    def add_2d(self, name, scaling_factor, loss_fn):
        self.losses_2d[name] = (scaling_factor, loss_fn)
        self.loss_weights_2d[name] = 1.0
        return self
    
    def remove_3d(self, name):
        if name in self.losses_3d:
            del self.losses_3d[name]
        if name in self.loss_weights_3d:
            del self.loss_weights_3d[name]
        return self

    def weight_3d(self, name, weight):
        self.loss_weights_3d[name] = weight
        return self
    
    def weight_2d(self, name, weight):
        self.loss_weights_2d[name] = weight
        return self
    
    def remove_2d(self, name):
        if name in self.losses_2d:
            del self.losses_2d[name]
        if name in self.loss_weights_2d:
            del self.loss_weights_2d[name]
        return self

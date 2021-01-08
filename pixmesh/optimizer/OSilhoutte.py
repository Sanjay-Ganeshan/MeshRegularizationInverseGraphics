from . import GenericOptimizer
import torch
import torch.nn

import pyredner as pyr

from ..common.guess import Guess
from ..common.target import Target

from ..loss.silhoutte import ImageSilhoutteLoss
from ..util.tensops import gpu

from .. import pxtyping as T

class OSilhoutte(GenericOptimizer):
    def __init__(self, initial_guess:Guess, weight, scale):
        super().__init__(initial_guess, "silhoutte", weight, scale)
        self.loss_fn = ImageSilhoutteLoss()
        self.loss_fn = self.loss_fn.to(pyr.get_device())
    def calculate_loss(self, guess: Guess, rendered_guess: T.Images, target: Target):
        if self.weight == 0.0 or self.scale == 0.0:
            return gpu([0])
        else:
            return self.scale * self.loss_fn(rendered_guess, target.images)

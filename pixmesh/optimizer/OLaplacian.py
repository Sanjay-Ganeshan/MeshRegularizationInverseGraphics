from . import GenericOptimizer
import torch
import torch.nn
import pyredner as pyr

from ..common.guess import Guess
from ..common.target import Target

from ..loss.laplacian import LaplacianLoss
from ..util.tensops import gpu

from .. import pxtyping as T

class OLaplacian(GenericOptimizer):
    def __init__(self, initial_guess: Guess, weight, scale, sos = False, sod = False, nel = False):
        super().__init__(initial_guess, "laplacian", weight, scale)
        self.loss_fn = LaplacianLoss(edge_length_scale=1 if nel else None,
                                     scale_by_orig_smoothness_div=sod,
                                     scale_by_orig_smoothness_sub=sos)
        self.loss_fn = self.loss_fn.to(pyr.get_device())
    def calculate_loss(self, guess: Guess, rendered_guess: T.Images, target: Target):
        if self.weight == 0.0 or self.scale == 0.0:
            return gpu([0])
        else:
            return self.scale * self.loss_fn(guess.mesh)

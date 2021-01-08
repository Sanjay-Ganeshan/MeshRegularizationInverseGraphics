from . import GenericRefiner
import torch
import pyredner as pyr

from ..common.guess import Guess

from ..util.meshops import laplacian

from .. import pxtyping as T
from ..external.manifold import manifold_upsample

class RManifoldUpsample(GenericRefiner):
    def __init__(self, enabled: bool, n_iter: T.Optional[int], upsample_factor: float, quality:int = 3000):
        super().__init__(enabled=enabled, n_iter=n_iter)
        self.upsample_factor = upsample_factor
        self.quality = quality

    
    def do_refinement(self, guess: Guess, config: T.ExperimentalConfiguration) -> T.Tuple[bool, Guess, T.ExperimentalConfiguration]:
        new_guess = manifold_upsample(guess, upsample_factor=self.upsample_factor, quality=self.quality)
        new_guess.mesh.vertices = laplacian(new_guess.mesh.vertices, new_guess.mesh.indices)
        new_config = config
        return (True, new_guess, new_config)

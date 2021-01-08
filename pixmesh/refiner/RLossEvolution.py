from . import GenericRefiner
import torch
import pyredner as pyr

from ..common.guess import Guess

from .. import pxtyping as T

class RLossEvolution(GenericRefiner):
    def __init__(self, enabled: bool, n_iter: T.Optional[int]):
        super().__init__(enabled=enabled, n_iter=n_iter)
    
    def do_refinement(self, guess: Guess, config: T.ExperimentalConfiguration) -> T.Tuple[bool, Guess, T.ExperimentalConfiguration]:
        if config.evolve_into is None:
            # Nothing to evolve into
            return (False, guess, config)
        else:
            # We can evolve!
            new_config = T.ExperimentalConfiguration.from_reduced_dict(config, config.evolve_into)
            new_guess = guess
            return (True, new_guess, new_config)

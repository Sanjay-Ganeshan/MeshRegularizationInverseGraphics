import itertools

from ..transformer import GuessTransformer, GenericMeshTransformer, GenericTextureTransformer
from ..common.guess import Guess
from ..common.target import Target
from ..common.groundtruth import GroundtruthExample
from ..common.scene_settings import SceneSettings

from .. import pxtyping as T
from ..util.tensops import gpu

class GenericOptimizer(object):
    def __init__(self, initial_guess: Guess, name, weight, scale):
        self.initial_guess = initial_guess
        self.name = name
        self.weight = weight
        self.scale = scale

    def calculate_loss(self, guess: Guess, rendered_guess: T.Images, target: Target):
        # Default loss is 0
        if self.weight == 0.0 or self.scale == 0:
            return gpu([0])
        else:
            return gpu([0])

class CombinedOptimizer(GenericOptimizer):
    def __init__(self, initial_guess: Guess, optims: T.List[GenericOptimizer]):
        super().__init__(initial_guess, "_total", weight=1.0, scale=1.0)
        self.optims = optims
        self.history = []
    def calculate_loss(self, guess: Guess, rendered_guess: T.Images, target: Target):
        each_loss = [
            (optim.name, optim.weight, optim.calculate_loss(guess, rendered_guess, target))
            for optim in self.optims
        ]

        total_loss = gpu([0])
        breakdown = {}
        for (name, weight, val) in each_loss:
            total_loss += weight * val
            breakdown[name] = val.item()
        breakdown[self.name] = total_loss.item()

        self.history.append(breakdown)
    
        return total_loss

    def unchanging(self, n_epochs, minchange):
        '''
        Returns true if it hasn't changed by more than minchange in n_epochs
        '''
        if len(self.history) < n_epochs:
            return False
        totals = [breakdown['_total'] for breakdown in self.history[-1*n_epochs:]]
        if max(totals) - min(totals) < minchange:
            return True

    def loss_increased(self):
        if len(self.history) < 2:
            return False
        else:
            return self.history[-1]['_total'] > self.history[-2]['_total']

    def get_last_breakdown(self):
        if len(self.history) > 0:
            return self.history[-1].copy()
        else:
            return {}
    
    def get_history(self):
        return [item.copy() for item in self.history]
    
    def set_prev_history(self, prev_history):
        self.history = [item.copy() for item in itertools.chain(prev_history, self.history)]

        


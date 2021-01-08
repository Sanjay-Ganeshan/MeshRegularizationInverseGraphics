from .target import Target
from .guess import Guess

from .. import pxtyping as T

class GroundtruthExample(object):
    '''
    A groundtruth example is a guess and a target
    '''
    def __init__(self, guess: Guess, target: Target):
        self.guess = guess
        self.target = target

from .. import GenericTextureTransformer
import pyredner as pyr
import torch
from ... import pxtyping as T
from ...common.guess import Guess
from ...util.imageops import copy_texture

class TNoChange(GenericTextureTransformer):
    def __init__(self, init_guess: Guess):
        super().__init__(init_guess)
    
    def get_transformed(self, texture: T.Optional[T.Texture]) -> T.Optional[T.Texture]:
        # Pass it raw
        return copy_texture(texture)

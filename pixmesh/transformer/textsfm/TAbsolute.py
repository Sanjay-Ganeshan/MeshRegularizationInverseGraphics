from .. import GenericTextureTransformer
import pyredner as pyr
import torch
from ... import pxtyping as T
from ...common.guess import Guess

class TAbsolute(GenericTextureTransformer):
    def __init__(self, init_guess: Guess):
        super().__init__(init_guess)
        if init_guess.texture is None:
            self.texels = None
        else:
            self.texels = init_guess.texture.texels.clone().detach()
            self.texels.requires_grad = True
    
    def get_transformed(self, texture: T.Optional[T.Texture]) -> T.Optional[T.Texture]:
        # Pass it raw
        if self.texels is None:
            return None
        else:
            return pyr.Texture(self.texels)
    
    def parameters(self):
        mine = [self.texels] if self.texels is not None else []
        return super().parameters() + mine

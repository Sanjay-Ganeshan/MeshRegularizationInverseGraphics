from .. import GenericTextureTransformer
import pyredner as pyr
import torch
import torch.nn
import torch.nn.init
from ... import pxtyping as T
from ...common.guess import Guess

class TResidual(GenericTextureTransformer):
    def __init__(self, init_guess: Guess):
        super().__init__(init_guess)
        if init_guess.texture is None:
            self.texels_r = None
        else:
            self.texels_r = torch.empty_like(init_guess.texture.texels)
            torch.nn.init.uniform_(self.texels_r, -0.001, 0.001)
            self.texels_r.requires_grad = True
    
    def get_transformed(self, texture: T.Optional[T.Texture]) -> T.Optional[T.Texture]:
        # Pass it raw
        if self.texels_r is None:
            return None
        else:
            return pyr.Texture(texture.texels + self.texels_r)
    
    def parameters(self):
        mine = [self.texels_r] if self.texels_r is not None else []
        return super().parameters() + mine

from ..common.guess import Guess
import pyredner as pyr
from .. import pxtyping as T

class GenericMeshTransformer(object):
    def __init__(self, init_guess: Guess):
        self.init_guess = init_guess
    def get_transformed(self, mesh: T.Mesh) -> T.Mesh:
        return mesh
    def parameters(self):
        return []

class GenericTextureTransformer(object):
    def __init__(self, init_guess: Guess):
        self.init_guess = init_guess
    def get_transformed(self, texture: T.Optional[T.Texture]) -> T.Optional[T.Texture]:
        if texture is None:
            return None
    def parameters(self):
        return []

class GuessTransformer(object):
    def __init__(self, mesh_transformer: GenericMeshTransformer, texture_transformer: GenericTextureTransformer):
        self.mesh_transformer = mesh_transformer
        self.texture_transformer = texture_transformer
    
    def get_transformed(self, guess: Guess):
        return Guess(self.mesh_transformer.get_transformed(guess.mesh), self.texture_transformer.get_transformed(guess.texture))

    def parameters(self):
        return self.mesh_transformer.parameters() + self.texture_transformer.parameters()

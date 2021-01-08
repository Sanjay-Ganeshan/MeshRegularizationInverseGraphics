import pyredner as pyr

from .. import pxtyping as T

class Guess(object):
    '''
    A guess contains a mesh and a texture
    '''
    def __init__(self, mesh: T.Mesh, texture:T.Optional[T.Texture]):
        self.mesh = mesh
        self.texture = texture



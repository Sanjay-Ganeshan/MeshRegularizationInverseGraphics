from .. import GenericMeshTransformer
import pyredner as pyr
import torch
from ... import pxtyping as T
from ...util.meshops import shallow_copy_mesh

class MNoChange(GenericMeshTransformer):
    def __init__(self, init_guess):
        super().__init__(init_guess)
    
    def get_transformed(self, mesh: T.Mesh) -> T.Mesh:
        # No change
        return shallow_copy_mesh(mesh)

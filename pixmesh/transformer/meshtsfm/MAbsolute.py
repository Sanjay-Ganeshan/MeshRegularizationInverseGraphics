from .. import GenericMeshTransformer
import pyredner as pyr
import torch
from ... import pxtyping as T
from ...util.meshops import shallow_copy_mesh

class MAbsolute(GenericMeshTransformer):
    def __init__(self, init_guess):
        super().__init__(init_guess)
        self.vert_positions = init_guess.mesh.vertices.clone().detach()
        self.vert_positions.requires_grad = True

    def parameters(self):
        return super().parameters() + [self.vert_positions]
    
    def get_transformed(self, mesh: T.Mesh) -> T.Mesh:
        # No change
        cp_mesh = shallow_copy_mesh(mesh)
        cp_mesh.vertices = self.vert_positions
        return cp_mesh

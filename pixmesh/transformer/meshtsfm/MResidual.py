from .. import GenericMeshTransformer
import pyredner as pyr
import torch
import torch.nn
import torch.nn.init
from ... import pxtyping as T
from ...common.guess import Guess
from ...util.meshops import shallow_copy_mesh

class MResidual(GenericMeshTransformer):
    def __init__(self, init_guess: Guess):
        super().__init__(init_guess)
        self.residual = torch.empty_like(init_guess.mesh.vertices)
        torch.nn.init.uniform_(self.residual, -0.001, 0.001)
        self.residual.requires_grad = True
    
    def get_transformed(self, mesh: T.Mesh) -> T.Mesh:
        # No change
        cp_mesh = shallow_copy_mesh(mesh)
        cp_mesh.vertices = mesh.vertices + self.residual
        return cp_mesh
    
    def parameters(self):
        return super().parameters() + [self.residual]

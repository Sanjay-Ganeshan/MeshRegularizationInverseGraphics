from .. import GenericMeshTransformer
import pyredner as pyr
import torch
import torch.nn
import torch.nn.init
from ... import pxtyping as T
from ...common.guess import Guess
from ...util.meshops import shallow_copy_mesh

class MNormal(GenericMeshTransformer):
    def __init__(self, init_guess: Guess):
        super().__init__(init_guess)
        self.normals = init_guess.mesh.normals.clone().detach()
        self.normal_mult = torch.zeros((len(self.normals),), dtype=torch.float32, device=pyr.get_device())
        torch.nn.init.uniform_(self.normal_mult, -0.001, 0.001)
        self.normal_mult.requires_grad = True
    
    def get_transformed(self, mesh: T.Mesh) -> T.Mesh:
        # No change
        cp_mesh = shallow_copy_mesh(mesh)
        cp_mesh.vertices = mesh.vertices + (self.normals * self.normal_mult.view((-1, 1)))
        return cp_mesh
    
    def parameters(self):
        return super().parameters() + [self.normal_mult]

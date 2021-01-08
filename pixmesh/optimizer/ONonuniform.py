from . import GenericOptimizer
import torch
import torch.nn
import pyredner as pyr

from ..common.guess import Guess
from ..common.target import Target

from ..util.tensops import gpu

from .. import pxtyping as T
from ..transformer.meshtsfm.neural.models.layers.mesh import Mesh as NeuralMesh




class ONonuniform(GenericOptimizer):
    def __init__(self, initial_guess: Guess, weight, scale):
        super().__init__(initial_guess, "nonuniform", weight, scale)
        # Use the NeuralMesh class to build the gfmm to interop with nonuniform loss
        neur_mesh = NeuralMesh("", hold_history=False, vs=initial_guess.mesh.vertices, faces=initial_guess.mesh.indices, device=pyr.get_device(), gfmm=True)
        self.gfmm = neur_mesh.gfmm

    def calculate_loss(self, guess: Guess, rendered_guess: T.Images, target: Target):
        if self.weight == 0.0 or self.scale == 0.0:
            return gpu([0])
        else:
            return self.scale * local_nonuniform_penalty(guess.mesh.vertices, guess.mesh.indices.long(), self.gfmm)


def local_nonuniform_penalty(vs, faces, gfmm):
    # non-uniform penalty
    # shape: (F,)
    per_face_area = mesh_area(vs, faces)

    # We want the average difference in area between every face and its neighbors
    # We ONLY need the faces for this, not the vertex positions
    diff = per_face_area[gfmm][:, 0:1] - per_face_area[gfmm][:, 1:]
    penalty = torch.norm(diff, dim=1, p=1)
    loss = penalty.sum() / penalty.numel()
    return loss

def mesh_area(vs, faces):
    v1 = vs[faces[:, 1]] - vs[faces[:, 0]]
    v2 = vs[faces[:, 2]] - vs[faces[:, 0]]
    area = torch.cross(v1, v2, dim=-1).norm(dim=-1)
    return area

from ... import GenericMeshTransformer
import pyredner as pyr
import torch
from .... import pxtyping as T
from ....common.guess import Guess
from ....dataloader import general as DL
from ....util import meshops

# Point2Mesh neural stuff

from .models.layers.mesh import Mesh, PartMesh
from .models.networks import init_net_no_optim
from .neuraloptions import Options as NOptions


class MNeural(GenericMeshTransformer):
    
    def __init__(self, init_guess: Guess):
        super().__init__(init_guess)
        self.opts = NOptions().args

        #with meshops.DiskObject(init_guess.mesh) as objfile:
        #    self.meshcnn_mesh = Mesh(objfile, hold_history=True, device=pyr.get_device())
        self.meshcnn_mesh = Mesh("", hold_history=True, device=pyr.get_device(), 
                                 vs=init_guess.mesh.vertices, faces=init_guess.mesh.indices)
        self.part_mesh = PartMesh(self.meshcnn_mesh, num_parts=1, bfs_depth=self.opts.overlap)
        self.net, self.rand_verts = init_net_no_optim(self.meshcnn_mesh, self.part_mesh, pyr.get_device(), self.opts)
        # Some requires_grad?
        
    def parameters(self):
        pars = super().parameters() + list(self.net.parameters())
        return pars
        
    
    def get_transformed(self, mesh: T.Mesh) -> T.Mesh:
        est_verts_l = list(self.net(self.rand_verts, self.part_mesh))
        assert len(est_verts_l) == 1, "Expected exactly 1 part"
        part_i = 0
        est_verts = est_verts_l[0]

        self.part_mesh.update_verts(est_verts[0], part_i)
        self.part_mesh.main_mesh.vs.detach_()

        # Now est_verts[0] is the new set of vertices

        cp_mesh = meshops.shallow_copy_mesh(mesh)
        # Change the vertices
        cp_mesh.vertices = est_verts[0]

        # No change
        return cp_mesh
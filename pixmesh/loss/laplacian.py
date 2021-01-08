import torch
import torch.nn as nn
import pyredner as pyr

import torch.nn.functional as F
from .. import pxtyping as T

def face_add(verts, faces, face_ix_from, face_ix_to, output, counts = None):
    data = verts[faces[:,face_ix_from]]
    face_add_arbitrary(faces, face_ix_to, data, output, counts)

def face_add_arbitrary(faces, face_ix_to, per_face_data, output, counts = None):
    ind_1d = faces[:,face_ix_to]
    one = torch.ones((1,), dtype=torch.float32, device=faces.device)
    if len(ind_1d.shape) < len(per_face_data.shape):
        ind_2d, _ = torch.broadcast_tensors(ind_1d.unsqueeze(1), per_face_data)
    else:
        ind_2d = ind_1d
    one_1d, _ = torch.broadcast_tensors(one, ind_1d)
    output.scatter_add_(0, ind_2d, per_face_data)
    if counts is not None:
        counts.scatter_add_(0, ind_1d, one_1d)

def laplacian(verts, faces):
    nvertices = verts.shape[0]
    nfaces = faces.shape[0]
    faces = faces.long().to(verts.device)
    totals = torch.zeros_like(verts, dtype=torch.float32, device=verts.device)
    num_neighbors = torch.zeros((nvertices,), dtype=torch.float32, device=verts.device)
    
    face_add(verts, faces, 1, 0, totals, num_neighbors)
    face_add(verts, faces, 2, 0, totals, num_neighbors)
    face_add(verts, faces, 0, 1, totals, num_neighbors)
    face_add(verts, faces, 2, 1, totals, num_neighbors)
    face_add(verts, faces, 0, 2, totals, num_neighbors)
    face_add(verts, faces, 1, 2, totals, num_neighbors)
    
    weighted_verts = totals / num_neighbors.unsqueeze(1)
    return weighted_verts

def average_edge_length_by_vertex(verts, faces):
    '''
    Get the average edge-length for all edges meeting at a given vertex
    '''
    faces = faces.long().to(verts.device)
    faces_t = faces.transpose(0, 1)

    edge1 = verts[faces_t[1]] - verts[faces_t[0]]
    edge2 = verts[faces_t[2]] - verts[faces_t[0]]
    edge3 = verts[faces_t[2]] - verts[faces_t[1]]

    edge1_l = torch.sqrt(torch.sum(edge1 * edge1, dim=1))
    edge2_l = torch.sqrt(torch.sum(edge2 * edge2, dim=1))
    edge3_l = torch.sqrt(torch.sum(edge3 * edge3, dim=1))

    nvertices = verts.shape[0]
    nfaces = faces.shape[0]
    totals = torch.zeros((nvertices,), dtype=torch.float32, device=verts.device)
    num_neighbors = torch.zeros((nvertices,), dtype=torch.float32, device=verts.device)

    face_add_arbitrary(faces, 0, edge1_l, totals, num_neighbors)
    face_add_arbitrary(faces, 1, edge1_l, totals, num_neighbors)
    face_add_arbitrary(faces, 0, edge2_l, totals, num_neighbors)
    face_add_arbitrary(faces, 2, edge2_l, totals, num_neighbors)
    face_add_arbitrary(faces, 1, edge3_l, totals, num_neighbors)
    face_add_arbitrary(faces, 2, edge3_l, totals, num_neighbors)

    avg_edge_l = totals / num_neighbors
    return avg_edge_l

def average_edge_length_by_faces(verts, faces):
    '''
    Get the average edge-length for each edge on every face
    '''

    faces = faces.long().to(verts.device).transpose(0, 1)

    edge1 = verts[faces[1]] - verts[faces[0]]
    edge2 = verts[faces[2]] - verts[faces[0]]
    edge3 = verts[faces[2]] - verts[faces[1]]

    edge1_l = torch.sqrt(torch.sum(edge1 * edge1, dim=1))
    edge2_l = torch.sqrt(torch.sum(edge2 * edge2, dim=1))
    edge3_l = torch.sqrt(torch.sum(edge3 * edge3, dim=1))

    avg_edge_l = edge1_l / 3 + edge2_l / 3 + edge3_l / 3

    return avg_edge_l

class LaplacianLoss(nn.Module):
    def __init__(self, edge_length_scale = None, scale_by_orig_smoothness_sub = False, scale_by_orig_smoothness_div = False):
        super().__init__()
        self.edge_length_scale = edge_length_scale

        assert (not (scale_by_orig_smoothness_div and scale_by_orig_smoothness_sub)), "Only one of LAP-SOS or LAP-SOD can be enabled at once"
        if scale_by_orig_smoothness_sub or scale_by_orig_smoothness_div:
            self.scale_fn = (lambda a,b: a-b) if scale_by_orig_smoothness_sub else (lambda a,b: a / b)
            self.scale_by_orig_smoothness = True
        else:
            self.scale_fn = None
            self.scale_by_orig_smoothness = False


        self.orig = None
    def forward(self, predicted_mesh):
        predicted_mesh_verts = predicted_mesh.vertices
        predicted_mesh_indices = predicted_mesh.indices
        lap = laplacian(predicted_mesh_verts, predicted_mesh_indices)
        # just measure squared distance from smooth version
        avg_edge_l = average_edge_length_by_vertex(predicted_mesh_verts, predicted_mesh_indices)
        avg_edge_l, _ = torch.broadcast_tensors(avg_edge_l.unsqueeze(1), lap)
        if self.edge_length_scale is not None:
            loss_per_vertex = (predicted_mesh_verts - lap) * self.edge_length_scale / avg_edge_l
        else:
            loss_per_vertex = predicted_mesh_verts - lap
        if self.scale_by_orig_smoothness:
            if self.orig is None:
                # Add a little bit so that nothing's / 0
                self.orig = loss_per_vertex.clone().detach() + 0.000001
            # Scale by orig. If the orig loss for a given vertex was HIGH,
            # it's not so bad (divide)
            # if the orig loss for a given vertex was LOW, then it is bad!
            loss_per_vertex = F.relu(self.scale_fn(loss_per_vertex, self.orig))

        sqrt_error = torch.sqrt(torch.sum(loss_per_vertex * loss_per_vertex, dim=1))
        mean_error = torch.mean(sqrt_error)
        return mean_error
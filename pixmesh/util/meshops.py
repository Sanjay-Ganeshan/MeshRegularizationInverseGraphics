import pyredner as pyr
import torch
import torch.cuda
import openmesh
from .voxlib import voxelize as vx
import tempfile
import os
import sys
import numpy as np

from .. import pxtyping as T

def voxelize(obj: T.Union[T.Mesh, str], verbose=False) -> T.VoxelGrid:
    # Now, let's read it into pyvista
    voxel_res=100
    voxel_grid = torch.zeros((voxel_res, voxel_res, voxel_res), device=pyr.get_device(), dtype=torch.float)
    with DiskObject(obj) as obj_fp:
        for (x,y,z) in vx.voxelize(obj_fp, 100):
            voxel_grid[x,y,z] = 1.0
    return voxel_grid

def voxel_iou(vox1: T.VoxelGrid, vox2: T.VoxelGrid) -> float:
    stacked = torch.stack([vox1, vox2], dim=0)
    intersection, _ = torch.min(stacked, dim=0)
    union, _ = torch.max(stacked, dim=0)
    intersecting_voxels = torch.sum(intersection)
    union_voxels = torch.sum(union)
    return (intersecting_voxels / union_voxels).item()

def decimate(obj: T.Mesh, factor=0.5, verbose = True):
    """
    Decimates a mesh, reducing the number of faces by 2.
    This is EXTREMELY inefficient, and not differentiable - use it sparingly!
    Modifies the input mesh.
    """

    with DiskObject(obj) as orig_out:
        # Now, let's load in the exported obj into openmesh
        mesh = openmesh.read_trimesh(orig_out)

    # Now, decimate by half
    orig_nfaces = mesh.n_faces()
    if verbose:
        print("Original # of faces:", orig_nfaces)
    decimator = openmesh.TriMeshDecimater(mesh)
    algorithm = openmesh.TriMeshModQuadricHandle()

    decimator.add(algorithm)
    decimator.initialize()
    decimator.decimate_to_faces(n_faces = round(orig_nfaces * factor))

    mesh.garbage_collection()

    if verbose:
        print("New # of faces:", mesh.n_faces())
    
    with DiskObject("decim.obj") as new_out:
        # Openmesh decimated version will get saved in its own dir
        openmesh.write_mesh(
            new_out,
            mesh)
        
        # Now, we have it. Load it back into redner
        decim_obj = pyr.load_obj(new_out, return_objects=True)[0]
    recompute_normals(decim_obj)

    return decim_obj

def subdivide(
    vertices: T.Union[torch.FloatTensor, torch.cuda.FloatTensor],
    indices: T.Union[torch.IntTensor, torch.cuda.IntTensor]):
    """
    Subdivides a mesh, increasing the number of vertices and faces
    :param vertices: The vertices of the original mesh. Shape |V| x 3
    :param indices: The faces of the mesh. Shape |F| x 3
    :returns: (new_vertices, new_faces)
    """
    pass

def laplacian(
    vertices: T.Union[torch.FloatTensor, torch.cuda.FloatTensor],
    indices: T.Union[torch.IntTensor, torch.cuda.IntTensor]):
    '''
    Smooth a mesh using the Laplacian method. Each output vertex becomes
    the average of its neighbors

    :param vertices: float32 tensor of vertices. (shape |V| x 3)
    :param indices: 
    :returns: this is a description of what is returned
    :raises keyError: raises an exception
    '''
    nvertices = vertices.shape[0]
    nfaces = indices.shape[0]
    indices = indices.long()
    totals = torch.zeros_like(vertices, dtype=torch.float32, device=vertices.device)
    num_neighbors = torch.zeros((nvertices,), dtype=torch.float32, device=vertices.device)
    
    _face_add(vertices, indices, 1, 0, totals, num_neighbors)
    _face_add(vertices, indices, 2, 0, totals, num_neighbors)
    _face_add(vertices, indices, 0, 1, totals, num_neighbors)
    _face_add(vertices, indices, 2, 1, totals, num_neighbors)
    _face_add(vertices, indices, 0, 2, totals, num_neighbors)
    _face_add(vertices, indices, 1, 2, totals, num_neighbors)
    
    weighted_vertices = totals / num_neighbors.unsqueeze(1)
    return weighted_vertices


def recompute_normals(obj: T.Mesh):
    """
    Recomputes smooth shading vertex normals for obj, and sets them
    accordingly.

    :param obj: A PyRedner object
    """
    obj.normals = pyr.compute_vertex_normal(obj.vertices.detach(), obj.indices.detach(), 'cotangent')
    obj.normal_indices = None

def normalize_to_box(obj: T.Mesh):
    # We want it to fit in a 2 x 2 x 2 box, centered at 0
    # Vertices is N x 3
    minXYZ, _ = torch.min(obj.vertices, dim=0)
    maxXYZ, _ = torch.max(obj.vertices, dim=0)
    rangeXYZ = maxXYZ - minXYZ
    # First, let's center to 0.
    halfRangeXYZ = rangeXYZ / 2.0
    # Move the corner to 0
    cornerAt0 = obj.vertices - minXYZ
    # Then subtract half the range
    centeredAt0 = cornerAt0 - halfRangeXYZ
    # Now divide by the biggest extent
    biggestExtent = torch.max(torch.abs(halfRangeXYZ))
    centeredAndScaled = centeredAt0 / biggestExtent
    # Now, let's detach replace the vertices
    obj.vertices = centeredAndScaled
    # Since the object's changed, we need to recompute normals
    recompute_normals(obj)

def shallow_copy_mesh(obj: T.Mesh):
    return pyr.Object(
        vertices=obj.vertices,
        indices=obj.indices,
        material=obj.material,
        light_intensity=obj.light_intensity,
        light_two_sided=obj.light_two_sided,
        uvs=obj.uvs,
        normals=obj.normals,
        uv_indices=obj.uv_indices,
        normal_indices=obj.normal_indices,
        colors=obj.colors
    )

class DiskObject():
    def __init__(self, obj: T.Union[T.Mesh, str, T.List[str]]):
        self.source = obj
        self.tmpdir = None
    
    def __enter__(self) -> str:
        is_pyr_obj = isinstance(self.source, pyr.Object)
        if is_pyr_obj:
            # Let's make a temporary directory
            self.tmpdir = os.path.abspath(tempfile.mkdtemp())
            obj_fp = os.path.join(self.tmpdir, "orig.obj")
            # And save the obj in that directory
            pyr.save_obj(self.source, obj_fp)
        else:
            # It's already a filepath. Does it exist?
            if isinstance(self.source, str) and os.path.exists(self.source):
                # Yeah, we'll be loading it from here
                obj_fp = os.path.abspath(self.source)
            else:
                # No. We want to output here.
                self.tmpdir = os.path.abspath(tempfile.mkdtemp())
                if isinstance(self.source, list):
                    obj_fp = [os.path.join(self.tmpdir, os.path.basename(src)) for src in self.source]
                elif isinstance(self.source, str):
                    obj_fp = os.path.join(self.tmpdir, os.path.basename(self.source))
                else:
                    raise ValueError(f"Unknown object type: {type(self.source)}. {self.source}")
        return obj_fp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tmpdir is not None:
            # Clean up the tmpdir we made
            for (root, subdirs, subfiles) in os.walk(self.tmpdir, topdown=False):
                # Always check if the path exists before removing it, as we're
                # removing dirs too!
                # Always guarenteed that it goes bottom-up
                for each_file_rel in subfiles:
                    each_file_abs = os.path.join(root, each_file_rel)
                    # Delete all the files
                    if os.path.exists(each_file_abs):
                        try:
                            os.remove(each_file_abs)
                        except:
                            print(f"Couldn't delete: {each_file_abs}", file=sys.stderr)
                # All the subdirs must be empty or non-existent
                for each_dir_rel in subdirs:
                    each_dir_abs = os.path.join(root, each_dir_rel)
                    if os.path.exists(each_dir_abs):
                        try:
                            os.rmdir(each_dir_abs)
                        except:
                            print(f"Couldn't delete: {each_dir_abs}", file=sys.stderr)
                # Now the directory is empty. Remove it
                if os.path.exists(root):
                    try:
                        os.rmdir(root)
                    except:
                        print(f"Couldn't delete: {root}", file=sys.stderr)
            if os.path.exists(self.tmpdir):
                print(f"Warning: Cleanup failed: {self.tmpdir}", file=sys.stderr)
        
# Helper functions to make these work
def _face_add(
    verts: T.Union[torch.FloatTensor, torch.cuda.FloatTensor],
    faces: T.Union[torch.LongTensor, torch.cuda.LongTensor],
    face_ix_from: int,
    face_ix_to: int, 
    output: T.Union[torch.FloatTensor, torch.cuda.FloatTensor],
    counts: T.Union[torch.FloatTensor, torch.cuda.FloatTensor]):
    """
    Helper function that adds the vertex values from the given face into
    an array, at its neighbors indices. Useful for Laplacians

    :param verts: The vertices of the mesh
    :param faces: The faces of the mesh
    :face_ix_from: Which index (0,1,2) of the faces to get from
    :face_ix_to: Which face index (0,1,2) to add at
    :output: Output for sum of vertices
    :counts: Output for # of elements added
    """
    ind_1d = faces[:,face_ix_to]
    one = torch.ones((1,), dtype=torch.float32, device=verts.device)
    data = verts[faces[:,face_ix_from]]
    ind_2d, _ = torch.broadcast_tensors(ind_1d.unsqueeze(1), data)
    one_1d, _ = torch.broadcast_tensors(one, ind_1d)
    output.scatter_add_(0, ind_2d, data)
    counts.scatter_add_(0, ind_1d, one_1d)

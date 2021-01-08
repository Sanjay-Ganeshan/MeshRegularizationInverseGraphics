'''
Runs the manifold program on the given input
'''

from ..dataloader import general as DL

from ..common.guess import Guess
from ..util.meshops import DiskObject
from ..util.imageops import copy_texture

from .. import pxtyping as T
import subprocess
import os

MANIFOLD_DIR = r'/home/ganesh/sanju/Manifold/build'  # path to manifold software (https://github.com/hjwdzh/Manifold)

SCRIPT_MANIFOLD_PATH = os.path.join(MANIFOLD_DIR, "manifold")
SCRIPT_SIMPLIFY_PATH = os.path.join(MANIFOLD_DIR, "simplify")

def manifold_upsample(guess: Guess, upsample_factor=1.0, quality=3000):
    # export before upsample
    n_faces_orig = len(guess.mesh.indices)
    n_faces_after = int(n_faces_orig * upsample_factor)

    with DiskObject(guess.mesh) as orig_save_path:
        with DiskObject(["watertight.obj", "simple.obj"]) as (watertight_fp, simple_fp):
            watertight_command = [SCRIPT_MANIFOLD_PATH, orig_save_path, watertight_fp, str(quality)]
            simplify_command = [SCRIPT_SIMPLIFY_PATH, "-i", watertight_fp, "-o", simple_fp, "-f", str(n_faces_after)]

            print("Executing>"," ".join(watertight_command))
            watertight_results = subprocess.run(watertight_command)
            assert watertight_results.returncode == 0, "Watertight failed"
            print("Executing>"," ".join(simplify_command))
            simplify_results = subprocess.run(simplify_command)
            assert simplify_results.returncode == 0, "Simplify failed"
            
            new_mesh = DL.load_mesh(simple_fp)
            # Set this new mesh
            new_guess = Guess(new_mesh, copy_texture(guess.texture))
        
    return new_guess
    

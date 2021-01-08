import os

import torch
import pyredner as pyr
import json

from . import general as DL
from ..common.guess import Guess
from ..common.groundtruth import GroundtruthExample
from ..common.scene_settings import SceneSettings
from ..common.target import Target

from ..renderer import GenericRenderer

from .. import pxtyping as T
from ..util.meshops import recompute_normals, normalize_to_box
from ..util.imageops import average_color_texture, noise_texture, copy_texture

def load_guess_and_gt(mesh_init_type: T.GuessInitializationType,
                      tex_init_type: T.GuessInitializationType,
                      gt_name: str,
                      gt_tex_mode: T.GroundtruthTexMode,
                      gt_camera_mode: T.GroundtruthCameraMode,
                      renderer: GenericRenderer,
                      guess_name: T.Optional[str] = None
                      ) -> T.Tuple[Guess, GroundtruthExample]:
    # First, let's load the groundtruth
    gt_dir = DL.get_groundtruth_dir(gt_name)

    # Groundtruth always has a mesh
    gt_mesh_fp = os.path.join(gt_dir, "mesh.obj")
    gt_tex_fp = os.path.join(gt_dir, "texture.png")

    if not os.path.isfile(gt_mesh_fp):
        raise FileNotFoundError(f"Mesh file not found: {gt_mesh_fp}")

    # Load the mesh
    gt_mesh = DL.load_mesh(gt_mesh_fp)
    # Normalize it
    normalize_to_box(gt_mesh)

    # Load the texture, as needed
    if gt_tex_mode == T.GroundtruthTexMode.DEFAULT:
        if not os.path.isfile(gt_tex_fp):
            raise FileNotFoundError(f"Texture file not found: {gt_tex_fp}")
        gt_tex = DL.load_texture(gt_tex_fp)
    elif gt_tex_mode == T.GroundtruthTexMode.NONE:
        gt_tex = None
    else:
        raise ValueError(f"Unknown tex mode: {gt_tex_mode}")

    # Now, load the camera
    if gt_camera_mode == T.GroundtruthCameraMode.SINGLE:
        gt_settings_fp = os.path.join(gt_dir, "scene_single.json")
    elif gt_camera_mode == T.GroundtruthCameraMode.REDUCED:
        gt_settings_fp = os.path.join(gt_dir, "scene_reduced.json")
    elif gt_camera_mode == T.GroundtruthCameraMode.DEFAULT:
        gt_settings_fp = os.path.join(gt_dir, "scene_default.json")
    elif gt_camera_mode == T.GroundtruthCameraMode.EXTRA:
        gt_settings_fp = os.path.join(gt_dir, "scene_extra.json")
    else:
        raise ValueError(f"Unknown camera mode: {gt_camera_mode}")
    
    if not os.path.isfile(gt_settings_fp):
        raise FileNotFoundError(f"Scene settings file not found: {gt_settings_fp}")

    gt_settings = DL.load_scene_settings(gt_settings_fp)

    # Now we have the groundtruth mesh, texture, and camera settings
    # Combine the mesh and texture into a "guess"
    gt_guess = Guess(gt_mesh, gt_tex)

    # Let's render target images from the gt guess
    with torch.no_grad():
        target_images = renderer.render(gt_guess, gt_settings)
    
    gt_target = Target(gt_settings, target_images)

    # Combine the target and guess into a groundtruth
    gt = GroundtruthExample(gt_guess, gt_target)
    
    # Where is the guess mesh?
    if mesh_init_type == T.GuessInitializationType.GENERIC:
        # Load the generic mesh
        guess_mesh_dir = DL.get_guess_dir('generic')
    elif mesh_init_type == T.GuessInitializationType.SPECIFIC:
        # Load the specific guess with the same name as gt
        guess_mesh_dir = DL.get_guess_dir(gt_name)
    elif mesh_init_type == T.GuessInitializationType.EXACT:
        # Load the mesh FROM GROUNDTRUTH folder
        guess_mesh_dir = DL.get_groundtruth_dir(gt_name)
    elif mesh_init_type == T.GuessInitializationType.OTHER:
        # Load an arbitrary guess by name
        guess_mesh_dir = DL.get_guess_dir(guess_name)
    else:
        raise ValueError(f"Unknown guess mesh initialization type: {mesh_init_type}")

    guess_mesh_fp = os.path.join(guess_mesh_dir, "mesh.obj")

    if not os.path.exists(guess_mesh_fp):
        raise FileNotFoundError(f"Could not find: {guess_mesh_fp}")

    # Load up the guess mesh
    guess_mesh = DL.load_mesh(guess_mesh_fp)

    # Normalize guess mesh
    normalize_to_box(guess_mesh)

    # Now, get the texture
    if gt_tex_mode == T.GroundtruthTexMode.NONE:
        guess_tex = None
    elif gt_tex_mode == T.GroundtruthTexMode.DEFAULT:
        if tex_init_type == T.GuessInitializationType.GENERIC:
            guess_tex = noise_texture()
        elif tex_init_type == T.GuessInitializationType.SPECIFIC:
            guess_tex = average_color_texture(target_images)
        elif tex_init_type == T.GuessInitializationType.EXACT:
            guess_tex = copy_texture(gt_tex)
        elif tex_init_type == T.GuessInitializationType.OTHER:
            guess_tex_fp = os.path.join(guess_mesh_dir, "texture.png")
            if not os.path.exists(guess_tex_fp):
                raise FileNotFoundError(f"Could not load OTHER texture: {guess_tex_fp}")
            guess_tex = DL.load_texture(guess_tex_fp)
        else:
            raise ValueError(f"Unknown guess texture initialization type: {tex_init_type}")
    else:
        raise ValueError("Unexpected GT tex mode")

    # Now we have both the guess mesh and texture, combine them.
    guess = Guess(guess_mesh, guess_tex)

    # Now we have the groundtruth and guess!
    return guess, gt

def load_experiment_config(experiment_name: str):
    default_exp_fp = DL.get_experiment_path("default")
    exp_fp = DL.get_experiment_path(experiment_name)
    parent_fp = os.path.join(os.path.dirname(exp_fp), "parent.json")
    
    with open(default_exp_fp) as stream:
        default_exp_js = json.load(stream)

    with open(exp_fp) as stream:
        this_exp_js = json.load(stream)

    if os.path.isfile(parent_fp):
        with open(parent_fp) as stream:
            parent_exp_js = json.load(stream)
    else:
        parent_exp_js = {}
    
    default_exp_js.update(parent_exp_js)
    default_exp_js.update(this_exp_js)

    assert (experiment_name == default_exp_js['experiment_name'] and os.path.splitext(os.path.basename(exp_fp))[0] == experiment_name), f"Experiment name does not match: {experiment_name}"

    return T.ExperimentalConfiguration.from_dict(default_exp_js)
    



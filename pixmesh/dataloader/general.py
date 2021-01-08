import os
import json

import pyredner as pyr

from .. import pxtyping as T
from ..common.scene_settings import SceneSettings
from ..common.guess import Guess
from ..common.groundtruth import GroundtruthExample
from ..common.target import Target
from ..util.tensops import gpu, gpui, cpu, cpui

import torch

def get_data_root():
    mydir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(os.path.dirname(mydir)), "data")
    return datadir

def find_file(filename, search_root):
    for (root, subdirs, subfiles) in os.walk(search_root, topdown=True):
        if filename in subfiles:
            return os.path.abspath(os.path.join(root, filename))
    return None

def get_experiment_path(exp_name: T.Optional[str] = None):
    exp_root = os.path.join(get_data_root(), "experiment_config")
    if exp_name is not None:
        exp_file = find_file(f"{exp_name}.json", exp_root)
        if exp_file is None:
            raise FileNotFoundError(f"Could not find experiment config: {exp_name}")
        return exp_file
    else:
        return exp_root

def get_groundtruth_dir(gt_name: T.Optional[str] = None):
    gt_root = os.path.join(get_data_root(), "groundtruth")
    if gt_name is not None:
        return os.path.join(gt_root, gt_name)
    else:
        return gt_root

def get_guess_dir(guess_name: str = None):
    guess_root = os.path.join(get_data_root(), "guess")
    if guess_name is not None:
        return os.path.join(guess_root, guess_name)
    else:
        return guess_root

def get_output_dir(exp_name:str = None):
    output_root = os.path.join(get_data_root(), "output")
    if exp_name is not None:
        return os.path.join(output_root, exp_name)
    else:
        return output_root

def load_texture(fp) -> T.Texture:
    '''
    Loads a texture, represented as a PNG
    '''
    img = pyr.imread(fp).to(pyr.get_device())
    tex = pyr.Texture(img)
    return tex

def load_mesh(fp) -> T.Mesh:
    obj = pyr.load_obj(fp, return_objects=True)[0]
    return obj

def save_mesh(mesh: T.Mesh, fp: str):
    pyr.save_obj(mesh, fp)

def save_texture(tex: T.Optional[T.Texture], fp: str):
    if tex is not None:
        pyr.imwrite(tex.texels.cpu().detach(), fp)

def save_guess(guess: Guess, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    mesh_fp = os.path.join(out_dir, "mesh.obj")
    tex_fp = os.path.join(out_dir, "texture.png")
    save_mesh(guess.mesh, mesh_fp)
    save_texture(guess.texture, tex_fp)

def load_scene_settings(fp) -> SceneSettings:
    with open(fp) as f:
        camera_settings = json.load(f)
 
    cameras = [
        pyr.Camera(
            position=cpu(camera_pos),
            look_at=cpu(single_or_list_get(camera_settings['look_at'], ix)),
            up=cpu(single_or_list_get(camera_settings['up'], ix)),
            fov=cpu(single_or_list_get(camera_settings['fov'], ix)),
            resolution=single_or_list_get(camera_settings['resolution'], ix),
            camera_type=pyr.camera_type.perspective
        )
        for ix, camera_pos in enumerate(camera_settings['positions'])
    ]

    def parse_one_light(dict_light):
        if dict_light['type'] == 'directional':
            return pyr.DirectionalLight(
                direction = gpu(dict_light['direction']),
                intensity = gpu(dict_light['intensity'])
            )
        else:
            return None
    
    lights = [
        parse_one_light(d) for d in camera_settings['lights']
    ]
    lights = [l for l in lights if l is not None]

    settings = SceneSettings(cameras, lights, raw=camera_settings)
    return settings

def single_or_list_get(item, index, expecting_list=True):
    is_single = False
    if isinstance(item, list):
        # This is a list. Are we expecting a list?
        if expecting_list:
            assert len(item) > 0, "Empty list"
            if isinstance(item[0], list):
                # This is an outer list
                is_single = False
            else:
                # This is the inner list
                is_single = True
        else:
            # This is a list
            is_single = False
    else:
        assert not expecting_list, "Got 1 item, expected a list"
        is_single = True
    
    if is_single:
        return item
    else:
        return item[index]

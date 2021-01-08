from .. import pxtyping as T
from ..common.scene_settings import SceneSettings
from ..common.guess import Guess
from . import GenericRenderer

import pyredner as pyr

class CoreRenderer(GenericRenderer):
    def __init__(self):
        super().__init__()

    def render(self, guess: Guess, scene_settings: SceneSettings, alpha=True) -> T.Images:
        # First, let's apply the guess' texture to the mesh
        mesh = guess.mesh

        if guess.texture is not None:
            mesh.material.diffuse_reflectance = guess.texture

        lights = scene_settings.lights

        # Now, let's make scenes
        scenes = [pyr.Scene(camera=c, objects=[mesh]) for c in scene_settings.cameras]
        imgs = pyr.render_deferred(scenes, lights=lights, alpha=alpha)
        return imgs


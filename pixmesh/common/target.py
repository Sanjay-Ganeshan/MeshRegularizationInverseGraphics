import pyredner as pyr
import torch

from .scene_settings import SceneSettings

from .. import pxtyping as T

class Target(object):
    '''
    A target contains scene parameters and images
    '''
    def __init__(self, scene_settings: SceneSettings, images: T.Optional[T.Images]):
        self.scene_settings = scene_settings
        self.images = images

from .. import pxtyping as T
from ..common.scene_settings import SceneSettings
from ..common.guess import Guess

class GenericRenderer(object):
    def __init__(self):
        pass

    def render(self, guess: Guess, scene_settings: SceneSettings) -> T.Images:
        pass

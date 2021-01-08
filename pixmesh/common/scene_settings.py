import pyredner as pyr
from .. import pxtyping as T

class SceneSettings(object):
    '''
    Scene settings consists of cameras and lights
    '''
    def __init__(self, cameras: T.CameraList, lights: T.LightList, raw: T.Optional[T.Any] = None):
        self.cameras = cameras
        self.lights = lights
        self.raw = raw

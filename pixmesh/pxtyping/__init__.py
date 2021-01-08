from typing import *
import torch
from enum import IntEnum
import pyredner as pyr

Images = torch.Tensor
Mesh = pyr.Object
Texture = pyr.Texture
Camera = pyr.Camera
Light = pyr.DeferredLight
VoxelGrid = torch.Tensor

CameraList = List[Camera]
LightList = List[Light]


''' Enums '''

class GuessInitializationType(IntEnum):
    GENERIC = 0
    SPECIFIC = 1
    EXACT = 2
    OTHER = 3

class GroundtruthTexMode(IntEnum):
    '''
    Should the groundtruth mesh be loaded with or without
    a texture?
    '''
    NONE = 0
    DEFAULT = 1

class GroundtruthCameraMode(IntEnum):
    '''
    How many views should the groundtruth be loaded with?
    '''
    SINGLE = 0
    REDUCED = 1
    DEFAULT = 2
    EXTRA = 3

class OptimizationAlgorithm(IntEnum):
    ADAM = 0
    SGD = 1

class MeshTransformerMode(IntEnum):
    NONE = 0
    ABSOLUTE = 1
    RESIDUAL = 2
    NORMAL = 3
    NEURAL = 4

class TexTransformerMode(IntEnum):
    NONE = 0
    ABSOLUTE = 1
    RESIDUAL = 2

def parse_possible_enum(key: str, val: Any) -> Any:
    key_to_enum = {
        "optimization_algorithm": OptimizationAlgorithm,
        "mesh_transformer": MeshTransformerMode,
        "tex_transformer": TexTransformerMode,
        "groundtruth_tex_mode": GroundtruthTexMode,
        "groundtruth_camera_mode": GroundtruthCameraMode,
        "guess_mesh_init": GuessInitializationType,
        "guess_tex_init": GuessInitializationType,
    }

    # No special parsing
    if key not in key_to_enum:
        return val

    # Might be an enum's name
    # Every enum val is in caps, underscores, or numbers
    is_enum_val = lambda s: all(map(lambda c: c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789', s))
    if isinstance(val, int):
        # It's already an int
        return val
    elif isinstance(val, str):
        # It might be an enumeration name
        enum_type = key_to_enum[key]
        if val in dir(enum_type) and is_enum_val(val):
            return getattr(enum_type, val)
        else:
            raise ValueError(f"{key}: {val} is an invalid enumeration")
    else:
        raise ValueError(f"{key}: {val} must be an enum value!")

def parse_enums_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: parse_possible_enum(k, d[k]) for k in d}

class ExperimentalConfiguration(object):
    def __init__(self,
            ## Constants

            # Gradient Descent Optimizer
            optimization_algorithm: OptimizationAlgorithm,

            # Dynamic LR
            lr_initial: float,
            lr_iters_for_minimum: int,
            lr_reduction_factor: float,
            lr_reductions_for_convergence: int,

            # Early stopping
            early_convergence_n_epochs: int,
            early_convergence_max_change: float,

            # Loss scales
            loss_scale_pyramid: float,
            loss_scale_silhoutte: float,
            loss_scale_laplacian: float,
            loss_scale_nonuniformity: float,
            loss_scale_flatness: float,

            # Refiner constants
            # None = upon convergence
            upsample_interval: Optional[int],
            upsample_factor: float,
            manifold_quality: int,
            # None = upon convergence
            evolution_interval: Optional[int],

            ## Variables

            # Transformer modes
            mesh_transformer: MeshTransformerMode,
            tex_transformer: TexTransformerMode,

            # Loss weights
            loss_weight_pyramid: float,
            loss_weight_silhoutte: float,
            loss_weight_laplacian: float,
            loss_weight_nonuniformity: float,
            loss_weight_flatness: float,

            # Loss configurations
            pyramid_num_levels: int,
            laplacian_scale_by_original_division: bool,
            laplacian_scale_by_original_subtraction: bool,
            laplacian_normalize_by_edge_length: bool,

            # Refiner config
            upsample_enabled: bool,
            manifold_enabled: bool,
            max_refinements: int,

            # Another experimental configuration
            evolve_into: Optional[Dict[str, Any]],

            # I / O
            experiment_name: str,
            groundtruth_name: str,
            groundtruth_tex_mode: GroundtruthTexMode,
            groundtruth_camera_mode: GroundtruthCameraMode,
            
            guess_mesh_init: GuessInitializationType,
            guess_tex_init: GuessInitializationType,
            guess_name: Optional[str]
        ):
        # Gradient Descent Optimizer
        self.optimization_algorithm = optimization_algorithm

        # Dynamic LR
        self.lr_initial = lr_initial
        self.lr_iters_for_minimum = lr_iters_for_minimum
        self.lr_reduction_factor = lr_reduction_factor
        self.lr_reductions_for_convergence = lr_reductions_for_convergence

        # Early stopping
        self.early_convergence_n_epochs = early_convergence_n_epochs
        self.early_convergence_max_change = early_convergence_max_change

        # Loss scales
        self.loss_scale_pyramid = loss_scale_pyramid
        self.loss_scale_silhoutte = loss_scale_silhoutte
        self.loss_scale_laplacian = loss_scale_laplacian
        self.loss_scale_nonuniformity = loss_scale_nonuniformity
        self.loss_scale_flatness = loss_scale_flatness

        # Refiner constants
        # None = upon convergence
        self.upsample_interval = upsample_interval
        self.upsample_factor = upsample_factor
        self.manifold_quality = manifold_quality
        # None = upon convergence
        self.evolution_interval = evolution_interval

        ## Variables

        # Transformer modes
        self.mesh_transformer = mesh_transformer
        self.tex_transformer = tex_transformer

        # Loss weights
        self.loss_weight_pyramid = loss_weight_pyramid
        self.loss_weight_silhoutte = loss_weight_silhoutte
        self.loss_weight_laplacian = loss_weight_laplacian
        self.loss_weight_nonuniformity = loss_weight_nonuniformity
        self.loss_weight_flatness = loss_weight_flatness

        # Loss configurations
        self.pyramid_num_levels = pyramid_num_levels
        self.laplacian_scale_by_original_division = laplacian_scale_by_original_division
        self.laplacian_scale_by_original_subtraction = laplacian_scale_by_original_subtraction
        self.laplacian_normalize_by_edge_length = laplacian_normalize_by_edge_length

        # Refiner config
        self.upsample_enabled = upsample_enabled
        self.manifold_enabled = manifold_enabled
        self.evolve_into = evolve_into
        self.max_refinements = max_refinements

        # I/O config
        self.experiment_name = experiment_name
        self.groundtruth_name = groundtruth_name
        self.groundtruth_tex_mode = groundtruth_tex_mode
        self.groundtruth_camera_mode = groundtruth_camera_mode
        
        self.guess_mesh_init = guess_mesh_init
        self.guess_tex_init = guess_tex_init
        self.guess_name = guess_name
        
        self.validate()

    def validate(self):
        invalid = False

        #invalid = invalid or self.groundtruth_camera_mode != GroundtruthCameraMode.DEFAULT
        invalid = invalid or self.mesh_transformer == MeshTransformerMode.NEURAL and self.groundtruth_name == "sine"
        #invalid = invalid or self.loss_weight_nonuniformity > 0
        #invalid = invalid or self.loss_weight_flatness > 0
        #invalid = invalid or (self.guess_mesh_init == GuessInitializationType.OTHER and self.guess_name not in "hiresgeneric")

        if invalid:
            raise NotImplementedError(f"{self.experiment_name} Used an option that has yet to be ported")

    def to_dict(self):
        return {
            "optimization_algorithm": self.optimization_algorithm,
            "lr_initial": self.lr_initial,
            "lr_iters_for_minimum": self.lr_iters_for_minimum,
            "lr_reduction_factor": self.lr_reduction_factor,
            "lr_reductions_for_convergence": self.lr_reductions_for_convergence,
            "early_convergence_n_epochs": self.early_convergence_n_epochs,
            "early_convergence_max_change": self.early_convergence_max_change,
            "loss_scale_pyramid": self.loss_scale_pyramid,
            "loss_scale_silhoutte": self.loss_scale_silhoutte,
            "loss_scale_laplacian": self.loss_scale_laplacian,
            "loss_scale_nonuniformity": self.loss_scale_nonuniformity,
            "loss_scale_flatness": self.loss_scale_flatness,
            "upsample_interval": self.upsample_interval,
            "upsample_factor": self.upsample_factor,
            "manifold_quality": self.manifold_quality,
            "evolution_interval": self.evolution_interval,
            "mesh_transformer": self.mesh_transformer,
            "tex_transformer": self.tex_transformer,
            "loss_weight_pyramid": self.loss_weight_pyramid,
            "loss_weight_silhoutte": self.loss_weight_silhoutte,
            "loss_weight_laplacian": self.loss_weight_laplacian,
            "loss_weight_nonuniformity": self.loss_weight_nonuniformity,
            "loss_weight_flatness": self.loss_weight_flatness,
            "pyramid_num_levels": self.pyramid_num_levels,
            "laplacian_scale_by_original_division": self.laplacian_scale_by_original_division,
            "laplacian_scale_by_original_subtraction": self.laplacian_scale_by_original_subtraction,
            "laplacian_normalize_by_edge_length": self.laplacian_normalize_by_edge_length,
            "upsample_enabled": self.upsample_enabled,
            "manifold_enabled": self.manifold_enabled,
            "max_refinements": self.max_refinements,
            "evolve_into": self.evolve_into,
            "experiment_name": self.experiment_name,
            "groundtruth_name": self.groundtruth_name,
            "groundtruth_tex_mode": self.groundtruth_tex_mode,
            "groundtruth_camera_mode": self.groundtruth_camera_mode,
            "guess_mesh_init": self.guess_mesh_init,
            "guess_tex_init": self.guess_tex_init,
            "guess_name": self.guess_name
        }
    
    @staticmethod
    def from_dict(d):
        d = parse_enums_in_dict(d)
        return ExperimentalConfiguration(**d)
    
    @staticmethod
    def from_reduced_dict(root_cfg, reduced_dict):
        d = root_cfg.to_dict()
        d.update(reduced_dict)
        return ExperimentalConfiguration.from_dict(d)

    def combine(self, child):
        d = self.to_dict()
        other_d = child.to_dict()
        d.update(other_d)
        return ExperimentalConfiguration.from_dict(d)
    
    def simplified_dict(self, root):
        mine = self.to_dict()
        root_d = root.to_dict()
        reduced_d = {}
        for each_key in mine:
            if each_key in root_d and mine[each_key] == root_d[each_key]:
                # We don't need this
                continue
            else:
                reduced_d[each_key] = mine[each_key]
        return reduced_d

class MetaConfiguration(object):
    def __init__(self,
        display_images:bool,
        print_loss_every:int,
        compare_images_every:int,
        save_mesh_every:int,
        max_iter_til_convergence: int,
        ):
        self.display_images = display_images
        self.print_loss_every = print_loss_every
        self.compare_images_every = compare_images_every
        self.save_mesh_every = save_mesh_every
        self.max_iter_til_convergence = max_iter_til_convergence
        
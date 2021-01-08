import json
from tqdm import tqdm
import os
import cv2
from ..dataloader.experiment import load_guess_and_gt, load_experiment_config

from ..dataloader import general as DL
from .. import pxtyping as T

# Guess / Target
from ..common.guess import Guess
from ..common.groundtruth import GroundtruthExample
from ..common.target import Target
from ..common.scene_settings import SceneSettings

# Transformer

# Base classes
from ..transformer import GuessTransformer, GenericMeshTransformer, GenericTextureTransformer

# Specific transformers
from ..transformer.meshtsfm.MNoChange import MNoChange
from ..transformer.meshtsfm.MAbsolute import MAbsolute
from ..transformer.meshtsfm.MResidual import MResidual
from ..transformer.meshtsfm.MNormal import MNormal
from ..transformer.meshtsfm.neural.MNeural import MNeural

from ..transformer.textsfm.TNoChange import TNoChange
from ..transformer.textsfm.TAbsolute import TAbsolute
from ..transformer.textsfm.TResidual import TResidual

# Renderer

# Base classes
from ..renderer import GenericRenderer

# Specific Renderers
from ..renderer.core import CoreRenderer

# Optimizer
from ..optimizer import GenericOptimizer, CombinedOptimizer

from ..optimizer.OFlatness import OFlatness
from ..optimizer.OLaplacian import OLaplacian
from ..optimizer.ONonuniform import ONonuniform
from ..optimizer.OPyramid import OPyramid
from ..optimizer.OSilhoutte import OSilhoutte

# Refiner

from ..refiner import GenericRefiner, CombinedRefiner

from ..refiner.RLossEvolution import RLossEvolution
from ..refiner.RManifoldUpsample import RManifoldUpsample

import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.nn.init
import pyredner as pyr
import sys

# UI usefulness
from  ..util import render as UI
from  ..util.meshops import DiskObject, voxelize, voxel_iou


class FullyLoadedExperiment(object):
    def __init__(self,
        init_guess: Guess,
        transformer: GuessTransformer,
        renderer: GenericRenderer,
        optimizer: CombinedOptimizer,
        refiner: GenericRefiner,
        groundtruth: GroundtruthExample,
        init_config: T.ExperimentalConfiguration
        ):
        self.init_guess = init_guess
        self.transformer = transformer
        self.renderer = renderer
        self.optimizer = optimizer
        self.refiner = refiner
        self.groundtruth = groundtruth
        self.init_config = init_config

def make_optimizer(mk_guess: Guess, mk_cfg: T.ExperimentalConfiguration) -> CombinedOptimizer:
    # 2D Losses
    inrLossPyramid = OPyramid(mk_guess, mk_cfg.loss_weight_pyramid, mk_cfg.loss_scale_pyramid, mk_cfg.pyramid_num_levels)
    inrLossSilhoutte = OSilhoutte(mk_guess, mk_cfg.loss_weight_silhoutte, mk_cfg.loss_scale_silhoutte)
    # 3D Losses
    inrLossLaplacian = OLaplacian(mk_guess, mk_cfg.loss_weight_laplacian, mk_cfg.loss_scale_laplacian, mk_cfg.laplacian_scale_by_original_subtraction, mk_cfg.laplacian_scale_by_original_division, mk_cfg.laplacian_normalize_by_edge_length)
    inrLossNonuniform = ONonuniform(mk_guess, mk_cfg.loss_weight_nonuniformity, mk_cfg.loss_scale_nonuniformity)
    inrLossFlatness = OFlatness(mk_guess, mk_cfg.loss_weight_flatness, mk_cfg.loss_scale_flatness)

    inrOptimizer = CombinedOptimizer(mk_guess, [inrLossPyramid, inrLossSilhoutte, inrLossLaplacian, inrLossNonuniform, inrLossFlatness])
    return inrOptimizer

def make_transformer(mk_guess: Guess, mk_cfg: T.ExperimentalConfiguration):
    '''
    Builds the transformer, so it can be rebuilt on refiner steps
    '''
    # Boot up the transformer
    # Get the right mesh transformer
    if mk_cfg.mesh_transformer == T.MeshTransformerMode.NONE:
        inrMeshTsfm = MNoChange(mk_guess)
    elif mk_cfg.mesh_transformer == T.MeshTransformerMode.ABSOLUTE:
        inrMeshTsfm = MAbsolute(mk_guess)
    elif mk_cfg.mesh_transformer == T.MeshTransformerMode.RESIDUAL:
        inrMeshTsfm = MResidual(mk_guess)
    elif mk_cfg.mesh_transformer == T.MeshTransformerMode.NORMAL:
        inrMeshTsfm = MNormal(mk_guess)
    elif mk_cfg.mesh_transformer == T.MeshTransformerMode.NEURAL:
        inrMeshTsfm = MNeural(mk_guess)
    else:
        raise ValueError(f"Unexpected mesh transformer: {mk_cfg.mesh_transformer}")

    # And the right texture transformer
    if mk_cfg.tex_transformer == T.TexTransformerMode.NONE:
        inrTexTsfm = TNoChange(mk_guess)
    elif mk_cfg.tex_transformer == T.TexTransformerMode.ABSOLUTE:
        inrTexTsfm = TAbsolute(mk_guess)
    elif mk_cfg.tex_transformer == T.TexTransformerMode.RESIDUAL:
        inrTexTsfm = TResidual(mk_guess)
    else:
        raise ValueError(f"Unexpected texture transformer: {mk_cfg.tex_transformer}")

    inrTransformer = GuessTransformer(inrMeshTsfm, inrTexTsfm)
    return inrTransformer

def experiment_exists(experiment_name):
    output_dir = DL.get_output_dir(experiment_name)
    if os.path.exists(os.path.join(output_dir, "guess_final", "mesh.obj")):
        return True
    else:
        return False

def evaluate_experiment(experiment_name):
    if not experiment_exists(experiment_name):
        # We don't have what we need
        print(f"Cannot evaluate {experiment_name}", file=sys.stderr)
    
    # We have what we need!
    output_dir = DL.get_output_dir(experiment_name)
    output_stat_path = os.path.join(output_dir, "metrics.json")

    if os.path.isfile(output_stat_path):
        return

    predicted_path = os.path.join(output_dir, "guess_final", "mesh.obj")
    predicted_tex_path = os.path.join(output_dir, "guess_final", "texture.png")
    cfg = load_experiment_config(experiment_name)
    mRenderer = CoreRenderer()
    mGuess, mGroundtruth = load_guess_and_gt(
        mesh_init_type=cfg.guess_mesh_init,
        tex_init_type=cfg.guess_tex_init,
        gt_name=cfg.groundtruth_name,
        gt_tex_mode=cfg.groundtruth_tex_mode,
        gt_camera_mode=cfg.groundtruth_camera_mode,
        renderer=mRenderer,
        guess_name=cfg.guess_name
    )
    # First, let's voxelize everything
    guess_vox = voxelize(mGuess.mesh)
    predicted_vox = voxelize(predicted_path)
    gt_vox = voxelize(mGroundtruth.guess.mesh)

    # Compute the IOU before and after
    before_iou = voxel_iou(guess_vox, gt_vox)
    after_iou = voxel_iou(predicted_vox, gt_vox)

    # Let's also compute image loss
    image_loss = OPyramid(mGuess, 1.0, 1.0, 1)
    
    
    predicted_mesh = DL.load_mesh(predicted_path)
    if os.path.exists(predicted_tex_path):
        predicted_tex = DL.load_texture(predicted_tex_path)
    else:
        predicted_tex = None

    predicted_guess = Guess(predicted_mesh, predicted_tex)

    rendered_initial = mRenderer.render(mGuess, mGroundtruth.target.scene_settings)
    rendered_prediction = mRenderer.render(predicted_guess, mGroundtruth.target.scene_settings)
    
    started_img_loss = image_loss.calculate_loss(mGuess, rendered_initial, mGroundtruth.target).item()
    ended_img_loss = image_loss.calculate_loss(predicted_guess, rendered_prediction, mGroundtruth.target).item()

    stats = {
        "initial": {
            "imageloss": started_img_loss,
            "iou": before_iou
        },
        "predicted": {
            "imageloss": ended_img_loss,
            "iou": after_iou
        }
    }

    # Now, let's find out how many iterations it took
    intermediate_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("guess_")]
    max_seen = {}
    for each_d in intermediate_dirs:
        spl = each_d.split("_")
        if len(spl) != 3:
            continue
        try:
            ref_stage, iter_num = int(spl[1]), int(spl[2])
            max_seen[ref_stage] = max(max_seen.get(ref_stage, 0), iter_num)
        except:
            continue
    
    max_seen["blank"] = 0
    total_iters = sum(max_seen.values())

    stats['niters'] = total_iters
    with open(output_stat_path, "w") as output_stat_fp:
        json.dump(stats, output_stat_fp, sort_keys=True, indent=4)
    
    return stats


def setup_experiment(experiment_name, skip_existing=True):
    if skip_existing and experiment_exists(experiment_name):
        return None
    cfg = load_experiment_config(experiment_name)
    # Let's set up the blocks first
    # Set up the renderer
    mRenderer = CoreRenderer()

    # Load the guess and gt
    mGuess, mGroundtruth = load_guess_and_gt(
        mesh_init_type=cfg.guess_mesh_init,
        tex_init_type=cfg.guess_tex_init,
        gt_name=cfg.groundtruth_name,
        gt_tex_mode=cfg.groundtruth_tex_mode,
        gt_camera_mode=cfg.groundtruth_camera_mode,
        renderer=mRenderer,
        guess_name=cfg.guess_name
    )

    # Build the transformer with the initial guess
    mTransformer = make_transformer(mGuess, cfg)
    # Now we have Guess, GT, Transformer, Renderer
    # Let's make the optimizer. It's the combination of a bunch of different ones.
    # Make each individually
    
    
    
    mOptimizer = make_optimizer(mGuess, cfg)

    # Finally, let's set up the refiner
    mRefinerManifold = RManifoldUpsample(cfg.manifold_enabled or cfg.upsample_enabled, cfg.upsample_interval, cfg.upsample_factor, cfg.manifold_quality)
    mRefinerEvolve = RLossEvolution(True, cfg.evolution_interval)
    mRefiner = CombinedRefiner([mRefinerManifold, mRefinerEvolve])

    # Now, we have all the pieces to run an experiment!
    exp = FullyLoadedExperiment(mGuess, mTransformer, mRenderer, mOptimizer, mRefiner, mGroundtruth, cfg)
    return exp
    
def print_and_get(prompt, val):
    print(prompt)
    return val

def run_experiment(experiment: FullyLoadedExperiment, meta_config: T.MetaConfiguration):
    # Set-up the output directory
    output_dir = DL.get_output_dir(experiment.init_config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll have a few outputs.
    outfp_loss_plot = os.path.join(output_dir, "loss_over_time.png")
    outfp_image_sequence = os.path.join(output_dir, "image_during_training.mp4")
    outfp_mesh_dir_template = os.path.join(output_dir, "guess_%s")

    out_imgseq = None

    with torch.no_grad():
        initial_images = experiment.renderer.render(experiment.init_guess, experiment.groundtruth.target.scene_settings)
    
    definitely_done = False

    # Set these once, in case for some reason max_refinements is 0
    mGuess = experiment.init_guess
    mTransformer = experiment.transformer
    mRenderer = experiment.renderer
    mOptimizer = experiment.optimizer
    mRefiner = experiment.refiner
    mGroundtruth = experiment.groundtruth
    mConfig = experiment.init_config
    comparison = None
    transformed_guess = mGuess

    for refinement_stage in range(experiment.init_config.max_refinements):
        mGuess = experiment.init_guess
        mTransformer = experiment.transformer
        mRenderer = experiment.renderer
        mOptimizer = experiment.optimizer
        mRefiner = experiment.refiner
        mGroundtruth = experiment.groundtruth
        mConfig = experiment.init_config

        transformed_guess = mGuess
        refinement_allowed = refinement_stage < experiment.init_config.max_refinements - 1


        # Setup optimizer and scheduler
        parameters = mTransformer.parameters()
        if mConfig.optimization_algorithm == T.OptimizationAlgorithm.ADAM:
            optimizer = torch.optim.Adam(parameters, lr=mConfig.lr_initial)
        elif mConfig.optimization_algorithm == T.OptimizationAlgorithm.SGD:
            optimizer = torch.optim.SGD(parameters, lr=mConfig.lr_initial)
        else:
            raise ValueError(f"Unknown optimization algorithm: {mConfig.optimization_algorithm}")
        #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda e_n: lr_mult if (e_n % lr_epochs_for_mult == 0 and e_n > 0 and e_n < lr_mult_until_epoch) else 1.0, last_epoch=-1)

        desired_mult = []
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda e_n: print_and_get("Reducing LR", desired_mult.pop(0)) if len(desired_mult) > 0 else 1.0, last_epoch=-1)
        nbad_in_row = 0
        nmults_used = 0

        for iternum in tqdm(range(meta_config.max_iter_til_convergence)):
            optimizer.zero_grad()
            transformed_guess = mTransformer.get_transformed(mGuess)
            rendered_guess = mRenderer.render(transformed_guess, mGroundtruth.target.scene_settings)
            total_loss = mOptimizer.calculate_loss(transformed_guess, rendered_guess, mGroundtruth.target)
        
            if mOptimizer.loss_increased():
                nbad_in_row += 1
                if nbad_in_row >= mConfig.lr_iters_for_minimum:
                    desired_mult.append(mConfig.lr_reduction_factor)
                    nbad_in_row = 0
                    nmults_used += 1

            total_loss.backward()

            # Zero out NANs in the gradient
            for each_param_group in optimizer.param_groups:
                for each_param in each_param_group['params']:
                    if each_param.grad is not None:
                        each_param.grad[torch.isnan(each_param.grad)] = 0.0

            optimizer.step()
            scheduler.step()

            just_converged = nmults_used >= mConfig.lr_reductions_for_convergence
            just_converged = just_converged or (mOptimizer.unchanging(mConfig.early_convergence_n_epochs, mConfig.early_convergence_max_change))

            # Calculate Refiner stuff. Don't use it yet though
            # If we're on the last refinement step, let it converge
            if refinement_allowed:
                # Refine the TRANSFORMED GUESS, not the original!
                needs_reinit, new_guess, new_config = mRefiner.refine(transformed_guess, mConfig, just_converged)
            else:
                needs_reinit = False
                new_guess = transformed_guess
                new_config = mConfig

            override_disp = needs_reinit or just_converged or iternum == meta_config.max_iter_til_convergence

            # Print loss every once in a while
            if override_disp or iternum % meta_config.print_loss_every == 0:
                print(mOptimizer.get_last_breakdown())

            # Draw a comparison image every once in a while        
            if override_disp or iternum % meta_config.compare_images_every == 0:
                comparison = UI.torch_to_np_image(UI.compare_images(initial_images, mGroundtruth.target.images, rendered_guess))
    
                if out_imgseq is None:
                    out_imgseq = UI.start_sequence(outfp_image_sequence, comparison)
                else:
                    UI.extend_sequence(out_imgseq, comparison)

                if meta_config.display_images:
                    cv2.imshow("compare", comparison)
                    
            
            if override_disp or iternum % meta_config.save_mesh_every == 0:
                outfp_this_guess = outfp_mesh_dir_template % f"{refinement_stage}_{iternum}"
                DL.save_guess(transformed_guess, outfp_this_guess)
                # We'll also output the loss history, just in case
                UI.plot_loss(mOptimizer.get_history(), save_to=outfp_loss_plot, show=False)
            
            if meta_config.display_images:
                keys = cv2.waitKey(16) & 0xFF
                if keys == ord('q'):
                    break
                if keys == ord('p'):
                    cv2.waitKey(0)

            if needs_reinit:
                previous_history = mOptimizer.get_history()
                modified_experiment = FullyLoadedExperiment(
                    init_guess=new_guess,
                    transformer=make_transformer(new_guess, new_config),
                    renderer=experiment.renderer,
                    optimizer=make_optimizer(new_guess, new_config),
                    refiner=experiment.refiner,
                    groundtruth=experiment.groundtruth,
                    init_config=new_config
                )

                modified_experiment.optimizer.set_prev_history(previous_history)
                
                # Set this to the var we look at
                experiment = modified_experiment

                # Done this refinement step. Break out of the iterations
                break
            elif just_converged:
                # We don't need a re-initialization, AND we've converged
                # If we have more refinement stages, reload from disk ;')

                if refinement_allowed:
                    with DiskObject(transformed_guess.mesh) as tmpondisk:
                        new_guess.mesh = DL.load_mesh(tmpondisk)
                    
                    previous_history = mOptimizer.get_history()
                    modified_experiment = FullyLoadedExperiment(
                        init_guess=new_guess,
                        transformer=make_transformer(new_guess, new_config),
                        renderer=experiment.renderer,
                        optimizer=make_optimizer(new_guess, new_config),
                        refiner=experiment.refiner,
                        groundtruth=experiment.groundtruth,
                        init_config=new_config
                    )

                    modified_experiment.optimizer.set_prev_history(previous_history)
                    
                    # Set this to the var we look at
                    experiment = modified_experiment

                break
        else:
            # It ran without converging at all!
            print("Warning: Did not converge! Increase your meta max_iter_til_convergence", file=sys.stderr)
            definitely_done = True
        #if definitely_done:
        #    break
        pass
    # At this point, we have the full loss history, image sequence, and final reconstruction
    UI.end_sequence(out_imgseq)
    UI.plot_loss(mOptimizer.get_history(), save_to=outfp_loss_plot, show=meta_config.display_images)
    outfp_this_guess = outfp_mesh_dir_template % f"final"
    DL.save_guess(transformed_guess, outfp_this_guess)

    if meta_config.display_images:
        if comparison is not None:
            UI.show_np_image(comparison)
        cv2.destroyAllWindows()

            




            







        

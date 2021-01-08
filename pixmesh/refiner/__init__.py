from ..common.guess import Guess
from ..common.groundtruth import GroundtruthExample
from ..common.scene_settings import SceneSettings
from ..common.target import Target

from .. import pxtyping as T

class GenericRefiner(object):
    def __init__(self, enabled: bool = False, n_iter: T.Optional[int] = None):
        self.current_iter = 0
        self.enabled = enabled
        self.max_iter = n_iter

    def refine(self, guess: Guess, config: T.ExperimentalConfiguration, just_converged: bool) -> T.Tuple[bool, Guess, T.ExperimentalConfiguration]:
        '''
        Given a guess, an experimental configuration, and a boolean flag determining if the optimizer just converged,
        Returns (needs_reinit, new_guess, new_config)

        If needs_reinit is True, the whole system will be reinitialized, with the given guess and config
        Otherwise, it can be ignored, and will just pass through the same guess and config it got
        '''
        if not self.enabled:
            # I'm not enabled, just pass through
            return (False, guess, config)
        else:
            # I'm enabled!
            self.current_iter += 1
            # Always activate if you converged
            should_activate = just_converged
            if self.max_iter is not None:
                # It's a number. Activate if the number of iterations has been reached. Ignore if max_iter < 0
                should_activate = should_activate or (self.max_iter >= 0 and self.current_iter >= self.max_iter)
            
            if should_activate:
                was_changed, new_guess, new_config = self.do_refinement(guess, config)
                # Reset the current iteration
                self.current_iter = 0
                return (was_changed, new_guess, new_config)

            else:
                return (False, guess, config)
        
    def do_refinement(self, guess: Guess, config: T.ExperimentalConfiguration) -> T.Tuple[bool, Guess, T.ExperimentalConfiguration]:
        '''
        Does the actual refinement procedure, returning the (changed, new guess, new experiment config)
        '''
        return False, guess, config

class CombinedRefiner(GenericRefiner):
    def __init__(self, refiners: T.List[GenericRefiner]):
        super().__init__(True, None)
        self.refiners = refiners

    def refine(self, guess: Guess, config: T.ExperimentalConfiguration, just_converged: bool) -> T.Tuple[bool, Guess, T.ExperimentalConfiguration]:
        # We'll entirely override the normal one
        c_guess = guess
        c_config = config
        c_just_converged = just_converged
        c_needs_reinit = False
        for each_refiner in self.refiners:
            needs_reinit, new_guess, new_config = each_refiner.refine(c_guess, c_config, c_just_converged)
            if needs_reinit:
                # This is new. But multiple might trigger in this pass!
                # So pass whatever the output of the PREVIOUSLY TRIGGERED REFINER was
                c_needs_reinit = True
                c_guess = new_guess
                c_config = new_config
        # Now all of them got to refine it!
        # Pass all the info back
        return (c_needs_reinit, c_guess, c_config)
    
    def do_refinement(self, guess: Guess, config: T.ExperimentalConfiguration) -> T.Tuple[bool, Guess, T.ExperimentalConfiguration]:
        return False, guess, config



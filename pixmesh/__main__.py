from .experiment import evaluate_experiment, setup_experiment, run_experiment, FullyLoadedExperiment
from . import pxtyping as T
import pyredner as pyr
import sys
import argparse

pyr.set_print_timing(False)

def main():
    meta = T.MetaConfiguration(
        display_images=True,
        print_loss_every=15,
        compare_images_every=10,
        save_mesh_every=80,
        max_iter_til_convergence=1200
    )

    evalmode = False

    experimentgroups = [
        ("VW", "A", 9),
        ("SM", "A", 4),
        ("SM", "B", 4),
        ("SM", "C", 4),
        ("SM", "D", 4),
        ("SM", "E", 4),
        ("SM", "F", 4),
        ("CF", "A", 4),
        ("IN", "A", 6),
        ("IN", "B", 4),
        ("IN", "C", 2),
        ("LP", "A", 4),
        ("LP", "B", 4),
        ("STX", "A", 4),
        ("MT", "A", 4),
        ("MT", "B", 4),
        ("MT", "C", 4),
        ("MT", "D", 4),
        ("MT", "E", 3), # Not enough VRAM for MT-E-3
        ("PY", "A", 4),
        ("PY", "B", 4),
        ("PY", "C", 4),
        ("SL", "A", 5),
        ("SL", "B", 4),
        #("TST","A", 5) #Try rendering all the meshes
    ]

    experimentgroups = [
        ("GN", "A", 1)
    ]

    to_skip = ["SM_E_0"]
    
    with open("errors.txt", "a") as err_f:
        for (q_name, group_name, n_exp) in experimentgroups:
            for exp_ix in range(n_exp):
                exp_name = f"{q_name}_{group_name}_{exp_ix}"
                if exp_name in to_skip:
                    continue
                if evalmode:
                    try:
                        print(exp_name, "\n", evaluate_experiment(exp_name))
                    except Exception as err:
                        err_s = f"Experiment {exp_name} didn't get evaluated ... {err}"
                        print(err_s, file=err_f)
                        print(err_s, file=sys.stderr)
                else:
                    try:
                        print(f"Setting up {exp_name}")
                        exp = setup_experiment(exp_name, skip_existing=True)
                        if exp is None:
                            print(f"Already have results for {exp_name}. Skipping.")
                            continue
                        print(f"Running {exp_name}")
                        run_experiment(exp, meta)
                    except Exception as err:
                        err_s = f"Experiment {exp_name} failed ... {err}"
                        print(err_s, file=err_f)
                        print(err_s, file=sys.stderr)

if __name__ == '__main__':
    main()
import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", nargs="+")
    exps = parser.parse_args().exp_name

    paths_to_view = []

    for expname in exps:
        outpath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output", expname, "image_during_training.mp4"))
        if os.path.exists(outpath):
            paths_to_view.append(outpath)
        else:
            print("DNE",outpath)
    
    if len(paths_to_view) == len(exps):
        subprocess.run(["vlc", *paths_to_view], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
if __name__ == "__main__":
    main()
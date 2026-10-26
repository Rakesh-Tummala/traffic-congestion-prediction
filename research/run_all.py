"""Run the parts of the research evaluation that need nothing but this repository and the internet.

    python research/run_all.py            # baselines, ablation, horizon, CV, bootstrap, conformal intervals
    python research/run_all.py --quick    # smoke test (a few minutes)

The camera logging, clip recording and labelling steps are interactive or long-running and are described in
research/README.md; so are the PeMS steps, which need a (free) PeMS account to download the ground-truth files.
Everything here is seeded, so a rerun reproduces research/results/*.csv up to the small run-to-run variation of
neural-network training (reported as a standard deviation over seeds).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def run(*args):
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run("research/baselines.py", *(["--quick"] if args.quick else []))
    run("research/conformal_intervals.py")
    print("\nDone. Tables: research/results/results.md. Next steps that need more than this repository: see research/README.md")


if __name__ == "__main__":
    main()

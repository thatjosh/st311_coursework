"""
Grid-search wrapper to run train.py over architectures and hyperparameter combinations on the RedBlueDoors-6x6-v0 environment.

To run this file:
python grid_search.py

To change hyperparams, please edit the variables in the script.
"""
import subprocess
import sys
import os
import itertools
import logging
import datetime

def main():
    log_filename = "grid_search.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting grid search")

    # Fixed environment and common settings
    env = "MiniGrid-RedBlueDoors-6x6-v0"
    seed = 7090
    algo = "ppo"
    save_interval = 10
    procs = 16
    frames = 200_000
    epochs = 4
    batch_size = 256
    frames_per_proc = None
    gae_lambda = 0.95
    max_grad_norm = 0.5
    optim_eps = 1e-8
    optim_alpha = 0.99
    clip_eps = 0.2
    recurrence = 4
    text = False

    # Hyperparams to try
    modtypes = ["mamba", "lstm", "transformer"]
    lrs = [1e-4, 1e-3, 1e-2]
    entropy_coefs = [0.1, 0.01]
    value_loss_coefs = [0.5, 1.0]
    discounts = [1, 0.99, 0.90]

    # Locate train.py relative to this script
    module_name = "scripts.train"

    # Iterate over all hyperparam combinations
    combinations = itertools.product(modtypes, lrs, entropy_coefs, value_loss_coefs, discounts)
    for idx, (mod, lr, ent_coef, val_coef, discount) in enumerate(combinations, start=1):

        start_time = datetime.datetime.now()
        logging.info(f"Run #{idx} starting at {start_time.isoformat()}")

        # Prefix model name with run index
        model_name = f"{idx}.{env}_{mod}_lr{lr}_ent{ent_coef}_vl{val_coef}_disc{discount}"

        cmd = [
            sys.executable, "-m", module_name,
            "--algo", algo,
            "--env", env,
            "--modtype", mod,
            "--model", model_name,
            "--seed", str(seed),
            "--save-interval", str(save_interval),
            "--procs", str(procs),
            "--frames", str(frames),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--discount", str(discount),
            "--lr", str(lr),
            "--gae-lambda", str(gae_lambda),
            "--entropy-coef", str(ent_coef),
            "--value-loss-coef", str(val_coef),
            "--max-grad-norm", str(max_grad_norm),
            "--optim-eps", str(optim_eps),
            "--optim-alpha", str(optim_alpha),
            "--clip-eps", str(clip_eps),
            "--recurrence", str(recurrence),
        ]
        if frames_per_proc is not None:
            cmd += ["--frames-per-proc", str(frames_per_proc)]
        if text:
            cmd.append("--text")

        print(f"Running #{idx}:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            logging.info(f"Run #{idx} finished at {end_time.isoformat()} (duration {duration}).")
        except subprocess.CalledProcessError as e:
            # log the failure and move on
            print(f"Run #{idx} failed (exit code {e.returncode}). Skipping.")
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            logging.warning(f"Run #{idx} failed at {end_time.isoformat()} (duration {duration}, exit code {e.returncode}). Continuing...")
            continue

if __name__ == "__main__":
    main()
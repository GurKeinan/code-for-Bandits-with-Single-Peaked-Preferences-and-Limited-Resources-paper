import sys
import os
import time
from multiprocessing import Pool
from functools import partial

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from instance_gen import generate_single_peaked_matrix
from model import BudgetedMatchingEnvironment
from mvm_algorithms import match_via_maximal
from config import *


# directories
results_dir = "mvm_results"
plots_dir   = "mvm_plots"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir,   exist_ok=True)

def process_instance(inst_id: int, n_runs: int, T: int, U: int, K: int,
                     results_dir: str, plots_dir: str):
    """
    Run N_RUNS of match_via_maximal on a single instance,
    save CSV & plot.
    """
    # reproducible per-instance RNG
    rng = np.random.RandomState(42 + inst_id + 5)  # change back to inst_id

    # generate a single-peaked reward matrix (do NOT shuffle columns)
    reward_matrix = generate_single_peaked_matrix(U, K, seed=42 + inst_id+5)  # change back to inst_id

    # container for regret trajectories
    regrets = np.zeros((n_runs, T))

    # inner progress bar over runs
    run_bar = tqdm(range(n_runs),
                   desc=f"Inst {inst_id} runs",
                   position=1,
                   leave=False,
                   file=sys.stdout)
    for run in run_bar:
        env = BudgetedMatchingEnvironment(U, K, reward_matrix, True)
        env.reset_random_state(1000 * (inst_id+5) + run) # change back to inst_id
        regrets[run] = match_via_maximal(env, T)
    run_bar.close()

    # compute stats
    mean = regrets.mean(axis=0)
    q5   = np.percentile(regrets,  5, axis=0)
    q95  = np.percentile(regrets, 95, axis=0)
    t    = np.arange(1, T + 1)

    # subsample ~1000 points for saving
    n_save = min(1000, T)
    idx    = np.linspace(0, T-1, n_save, dtype=int)

    # log-transform and fit
    log_t      = np.log(t)
    log_mean   = np.log(mean)
    log_q5     = np.log(q5)
    log_q95    = np.log(q95)
    slope, intercept = np.polyfit(log_t, log_mean, 1)
    log_pred   = slope * log_t + intercept

    # save to CSV
    df = pd.DataFrame({
        't':                t[idx],
        'log_t':            log_t[idx],
        'mean_regret':      mean[idx],
        'log_mean_regret':  log_mean[idx],
        'q5_regret':        q5[idx],
        'log_q5_regret':    log_q5[idx],
        'q95_regret':       q95[idx],
        'log_q95_regret':   log_q95[idx],
        'fitted_log_regret': log_pred[idx],
    })
    df.attrs = {
        'slope':      slope,
        'intercept':  intercept,
        'U':          U,
        'K':          K,
        'T':          T,
        'N_RUNS':     n_runs,
        'instance':   inst_id
    }
    df.to_csv(os.path.join(results_dir, f'mvm_regret_results_instance_{inst_id+5}.csv'),
              index=False) # change back to without +5

    # make and save the plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(log_t, log_q5, log_q95, alpha=0.3, label="5th–95th percentile")
    plt.plot(log_t, log_mean,    linewidth=2, label="Mean Log Regret")
    plt.plot(log_t, log_pred,
             linestyle='--', linewidth=2, label=f"Fit: slope={slope:.2f}")
    plt.xlabel("log(round $t$)")
    plt.ylabel("log(cumulative regret)")
    plt.title(f"Instance {inst_id}: Log–Log Regret Growth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'instance_{inst_id+5}_regret_plot.png')) # change back to without +5
    plt.close()

    return inst_id

if __name__ == "__main__":
    start = time.time()

    worker = partial(process_instance,
                     n_runs=N_RUNS,
                     T=T_MvM,
                     U=U,
                     K=K,
                     results_dir=results_dir,
                     plots_dir=plots_dir)

    # top-level progress bar over instances
    with Pool(processes=5) as pool:
        for _ in tqdm(pool.imap_unordered(worker, range(N_INSTANCES)),
                      total=N_INSTANCES,
                      desc="Instances",
                      position=0,
                      file=sys.stdout):
            pass

    elapsed = time.time() - start
    print(f"All {N_INSTANCES} instances done in {elapsed:.1f}s.")
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from instance_gen import generate_single_peaked_matrix
from model import BudgetedMatchingEnvironment
from emc_algorithms import explore_then_commit
from config import *

# dir to save outputs
CSV_DIR = "emc_csvs"
PLOT_DIR = "emc_plots"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def process_instance(inst: int):
    """
    Worker function: runs all N_RUNS for a single instance,
    saves the two CSVs and a plot for this instance.
    """
    # reproducible per-instance
    rng = np.random.RandomState(47 + inst)

    regrets = np.zeros((N_RUNS, len(T_EMC_LIST)))
    # inner progress bar for runs
    run_bar = tqdm(range(N_RUNS),
                   desc=f"Inst {inst} runs",
                   position=1,
                   leave=False,
                   file=sys.stdout)
    for run in run_bar:
        # generate and shuffle reward matrix
        reward_matrix = generate_single_peaked_matrix(U, K, seed=47 + inst)
        reward_matrix = rng.permutation(reward_matrix.T).T

        env = BudgetedMatchingEnvironment(U, K, reward_matrix, False)
        for t_idx, T in enumerate(T_EMC_LIST):
            env.reset_random_state(1000 * inst + run)
            regrets[run, t_idx] = explore_then_commit(env, T)
    run_bar.close()

    # compute summary stats
    mean = regrets.mean(axis=0)
    q5 = np.percentile(regrets, 5, axis=0)
    q95 = np.percentile(regrets, 95, axis=0)
    t = np.array(T_EMC_LIST)
    log_t = np.log(t)
    log_mean = np.log(mean)
    log_q5 = np.log(q5)
    log_q95 = np.log(q95)
    slope, intercept = np.polyfit(log_t, log_mean, 1)
    log_y_pred = slope * log_t + intercept

    # save summary CSV
    results_df = pd.DataFrame({
        'T': t,
        'log_T': log_t,
        'mean_regret': mean,
        'log_mean_regret': log_mean,
        'q5_regret': q5,
        'log_q5_regret': log_q5,
        'q95_regret': q95,
        'log_q95_regret': log_q95,
        'fitted_log_regret': log_y_pred
    })
    results_df.attrs = {
        'slope': slope,
        'intercept': intercept,
        'U': U,
        'K': K,
        'N_RUNS': N_RUNS,
        'instance': inst
    }
    results_df.to_csv(os.path.join(
        CSV_DIR, f'regret_results_instance_{inst}.csv'), index=False)

    # save raw regrets CSV
    raw_regrets_df = pd.DataFrame(regrets, columns=[f'T_{t}' for t in T_EMC_LIST])
    raw_regrets_df.to_csv(os.path.join(
        CSV_DIR, f'raw_regrets_instance_{inst}.csv'), index=False)

    # create and save plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(log_t, log_q5, log_q95, alpha=0.3,
                     label="5th-95th percentile")
    plt.plot(log_t, log_mean, 'b-', linewidth=2, label="Mean Log Regret")
    plt.plot(log_t, log_y_pred, 'r--', linewidth=2,
             label=fr"Fit: slope={slope:.2f}")
    plt.xlabel("log(round $t$)")
    plt.ylabel("log(cumulative regret)")
    plt.title(f"Instance {inst}: Log-Log Regret Growth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'instance_{inst}_regret_plot.png'))
    plt.close()

    return inst  # just to signal completion


if __name__ == "__main__":
    start_time = time.time()
    cpu_num = min(cpu_count(), 5)  # limit to 8 cores for stability

    # process all instances in parallel
    with Pool(processes=cpu_num) as pool:
        # tqdm over pool.imap to get instance-level progress bar
        for _ in tqdm(pool.imap_unordered(process_instance, range(N_INSTANCES)),
                      total=N_INSTANCES,
                      desc="Instances",
                      position=0,
                      file=sys.stdout):
            pass

    total_time = time.time() - start_time
    print(f"All instances completed in {total_time:.1f} seconds.")

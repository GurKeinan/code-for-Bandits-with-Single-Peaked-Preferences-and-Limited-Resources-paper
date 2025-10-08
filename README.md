# Simulations for Budgeted Matching with Single-Peaked Preferences

This repository contains code for simulating and analyzing algorithms for budgeted matching problems with single-peaked user preferences. The project includes implementations of algorithms, data generation, plotting utilities, and scripts for running large-scale experiments in parallel.

## Table of Contents

- [Project Structure](#project-structure)
- [Main Files](#main-files)
- [Algorithms](#algorithms)
- [Plotting](#plotting)
- [Data Generation](#data-generation)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [References](#references)

---

## Project Structure

```
.
├── config.py
├── emc_algorithms.py
├── emc_plotting.py
├── extract_order_c1p_wrapper.py
├── instance_gen.py
├── main_emc_parallelized.py
├── main_mvm_parallelized.py
├── model.py
├── mvm_algorithms.py
├── mvm_plotting.py
├── sagemath_c1p.py
├── sp_matching.py
├── utils.py
├── emc_csvs/
├── emc_plots/
├── mvm_results/
├── mvm_plots/
└── README.md
```

---

## Main Files

- **main_emc_parallelized.py**: Runs the Explore-then-Commit (EMC) algorithm in parallel across multiple instances and saves results/plots.
- **main_mvm_parallelized.py**: Runs the Match-via-Maximal (MvM) algorithm in parallel and saves results/plots.
- **config.py**: Central configuration for experiment parameters (number of users, arms, runs, etc.).
- **instance_gen.py**: Generates single-peaked preference matrices for users.
- **model.py**: Defines the `BudgetedMatchingEnvironment` class, which simulates the environment for the algorithms.
- **emc_algorithms.py**: Implements the Explore-then-Commit algorithm and related utilities.
- **mvm_algorithms.py**: Implements the Match-via-Maximal algorithm and regret computation.
- **sp_matching.py**: Dynamic programming algorithm for optimal matching under single-peaked preferences.
- **extract_order_c1p_wrapper.py**: Extracts an ordering of items to make a matrix approximately single-peaked.
- **utils.py**: Utility functions, including verification of single-peakedness.
- **emc_plotting.py**: Plotting utilities for EMC results.
- **mvm_plotting.py**: Plotting utilities for MvM results.
- **sagemath_c1p.py**: Consecutive ones property test (used for ordering extraction).

---

## Algorithms

- **Explore-then-Commit-and-Match (EMC)**: An algorithm that explores for a fixed period, then commits to the best matching found.
- **Match-via-Maximal (MvM)**: An algorithm that repeatedly matches users to arms using a maximal matching strategy.
- **Single-Peaked Matching**: Uses dynamic programming to find optimal matchings when user preferences are single-peaked.

---

## Plotting

- **emc_plotting.py** and **mvm_plotting.py** provide classes to load results and generate regret plots for each instance and aggregate statistics.

---

## Data Generation

- **instance_gen.py**: Generates random single-peaked matrices for user preferences, ensuring each user has a unique peak.

---

## Configuration

- All experiment parameters (number of users, arms, runs, time horizons, plot settings) are set in `config.py` for easy modification.

---

## Dependencies

Install the following Python libraries (preferably in a virtual environment):

- numpy
- pandas
- matplotlib
- tqdm

You can install them with:

```bash
pip install numpy pandas matplotlib tqdm
```

Some algorithms may require additional packages (e.g., SageMath for `sagemath_c1p.py`). If so, follow the instructions in that file or install [SageMath](https://www.sagemath.org/).

---

## How to Run

1. **Run EMC Experiments:**

   ```bash
   python main_emc_parallelized.py
   ```

   This will generate results in `emc_csvs/` and plots in `emc_plots/`.

2. **Run MvM Experiments:**

   ```bash
   python main_mvm_parallelized.py
   ```

   This will generate results in `mvm_results/` and plots in `mvm_plots/`.

3. **Plotting:**

   Use the plotting classes in `emc_plotting.py` and `mvm_plotting.py` to generate or customize plots.

---

## Outputs

- **CSV files**: Regret results for each instance and algorithm.
- **PNG and PDF plots**: Regret curves for each instance and aggregate statistics.
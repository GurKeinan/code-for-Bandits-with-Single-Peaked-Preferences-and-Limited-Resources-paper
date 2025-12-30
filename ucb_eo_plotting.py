import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re

from config import *

# Plotting configuration
plt.style.use('bmh')

class UCBEOResultsPlotter:
    def __init__(self, results_dir="ucb_eo_results", plots_dir="ucb_eo_plots"):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def load_results(self, instance_ids=None):
        """Load results from CSV files."""
        results = {}
        csv_pattern = os.path.join(self.results_dir, 'ucb_eo_regret_results_instance_*.csv')
        csv_files = glob.glob(csv_pattern)

        available_instances = []
        for csv_file in csv_files:
            match = re.search(r'instance_(\d+)\.csv', csv_file)
            if match:
                available_instances.append(int(match.group(1)))

        available_instances.sort()

        if instance_ids is None:
            instance_ids = available_instances
        else:
            missing = set(instance_ids) - set(available_instances)
            if missing:
                print(f"Warning: Instances {missing} not found. Available: {available_instances}")
            instance_ids = [i for i in instance_ids if i in available_instances]

        for inst_id in instance_ids:
            csv_file = os.path.join(self.results_dir, f'ucb_eo_regret_results_instance_{inst_id}.csv')
            try:
                df = pd.read_csv(csv_file)
                if len(df) >= 2:
                    log_t = df['log_t'].values
                    fitted_log = df['fitted_log_regret'].values
                    slope = (fitted_log[-1] - fitted_log[0]) / (log_t[-1] - log_t[0])
                else:
                    slope = np.nan

                metadata = {'slope': slope, 'instance': inst_id}
                results[inst_id] = (df, metadata)
            except Exception as e:
                print(f"Error loading instance {inst_id}: {e}")

        return results

    def plot_mean_with_std(self, instance_ids=None, save_individual=False,
                          show_reference_line=True, reference_slope=0.5,
                          reference_intercept=None, auto_intercept_offset=0.5):
        """Plot mean regret curve across all instances with standard deviation area."""
        results = self.load_results(instance_ids)

        if not results:
            print("No results to plot!")
            return

        all_log_t = []
        all_regret_data = []

        for inst_id, (df, metadata) in results.items():
            all_log_t.extend(df['log_t'].values)
            all_regret_data.append(df)

        min_log_t = max([df['log_t'].min() for df in all_regret_data])
        max_log_t = min([df['log_t'].max() for df in all_regret_data])

        n_points = min([len(df) for df in all_regret_data])
        common_log_t = np.linspace(min_log_t, max_log_t, n_points)

        regret_matrix = np.zeros((len(results), n_points))

        for i, (inst_id, (df, metadata)) in enumerate(results.items()):
            regret_interp = np.interp(common_log_t, df['log_t'].values, df['log_mean_regret'].values)
            regret_matrix[i, :] = regret_interp

        mean_regret = np.mean(regret_matrix, axis=0)
        std_regret = np.std(regret_matrix, axis=0)

        plt.figure(figsize=(12, 8))

        if show_reference_line:
            if reference_intercept is None:
                max_regret = np.max(mean_regret + std_regret)
                ref_intercept = max_regret + auto_intercept_offset
            else:
                ref_intercept = reference_intercept

            ref_log_regret = reference_slope * common_log_t + ref_intercept

            if reference_slope == 0.5:
                ref_label = r'$\sqrt{T}$ growth'
            else:
                ref_label = f'slope = {reference_slope}'

            plt.plot(common_log_t, ref_log_regret,
                    color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle=':', alpha=0.8,
                    label=ref_label)

        plt.fill_between(common_log_t,
                        mean_regret - std_regret,
                        mean_regret + std_regret,
                        alpha=0.3, color='blue', label='±1 std')

        plt.plot(common_log_t, mean_regret,
                color='blue', linewidth=MAIN_LINE_WIDTH,
                label=f'Mean across {len(results)} instances')

        plt.xlabel('log(round $t$)', fontsize=LABEL_FONTSIZE)
        plt.ylabel('log(cumulative regret)', fontsize=LABEL_FONTSIZE)
        plt.title('UCB-Extract-Order: Log-Log Regret Growth', fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()

        combined_filename = f"ucb_eo_mean_std_instances_{'_'.join(map(str, results.keys()))}.pdf"
        plt.savefig(os.path.join(self.plots_dir, combined_filename), dpi=300, bbox_inches='tight')

        plt.show()

    def plot_individual_instances(self, instance_ids=None, show_slopes=False,
                                show_confidence=True, save_individual=True,
                                show_reference_line=True, reference_slope=0.5,
                                reference_intercept=None, auto_intercept_offset=0.5):
        """Plot individual log-log regret curves for each instance."""
        results = self.load_results(instance_ids)

        if not results:
            print("No results to plot!")
            return

        colors = cm.tab10(np.linspace(0, 1, len(results)))

        plt.figure(figsize=(12, 8))

        all_log_t = []
        for inst_id, (df, metadata) in results.items():
            all_log_t.extend(df['log_t'].values)

        min_log_t = min(all_log_t)
        max_log_t = max(all_log_t)

        if show_reference_line:
            ref_log_t = np.linspace(min_log_t, max_log_t, 100)

            if reference_intercept is None:
                max_regret = max([df['log_mean_regret'].max() for _, (df, _) in results.items()])
                ref_intercept = max_regret + auto_intercept_offset
            else:
                ref_intercept = reference_intercept

            ref_log_regret = reference_slope * ref_log_t + ref_intercept

            if reference_slope == 0.5:
                ref_label = r'$\sqrt{T}$ growth'
            else:
                ref_label = f'slope = {reference_slope}'

            plt.plot(ref_log_t, ref_log_regret,
                    color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle=':', alpha=0.8,
                    label=ref_label)

        for i, (inst_id, (df, metadata)) in enumerate(results.items()):
            color = colors[i]

            if show_confidence and 'log_q5_regret' in df.columns and 'log_q95_regret' in df.columns:
                plt.fill_between(df['log_t'], df['log_q5_regret'], df['log_q95_regret'],
                               alpha=0.2, color=color)

            plt.plot(df['log_t'], df['log_mean_regret'],
                           color=color, linewidth=MAIN_LINE_WIDTH,
                           label=f'Instance {inst_id}')

        plt.xlabel('log(round $t$)', fontsize=LABEL_FONTSIZE)
        plt.ylabel('log(cumulative regret)', fontsize=LABEL_FONTSIZE)
        plt.title('UCB-Extract-Order: Individual Instances', fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()

        combined_filename = f"ucb_eo_combined_instances_{'_'.join(map(str, results.keys()))}.pdf"
        plt.savefig(os.path.join(self.plots_dir, combined_filename), dpi=300, bbox_inches='tight')

        plt.show()

    def plot_with_reference(self, instance_ids=None, reference_slope=0.5,
                          reference_intercept=None, auto_intercept_offset=0.5,
                          show_mean_std=True):
        """Convenience method to plot with reference line."""
        if show_mean_std:
            self.plot_mean_with_std(
                instance_ids=instance_ids,
                show_reference_line=True,
                reference_slope=reference_slope,
                reference_intercept=reference_intercept,
                auto_intercept_offset=auto_intercept_offset
            )
        else:
            self.plot_individual_instances(
                instance_ids=instance_ids,
                show_slopes=False,
                show_confidence=True,
                show_reference_line=True,
                reference_slope=reference_slope,
                reference_intercept=reference_intercept,
                auto_intercept_offset=auto_intercept_offset
            )

    def summary_statistics(self, instance_ids=None):
        """Print summary statistics of the results."""
        results = self.load_results(instance_ids)

        if not results:
            print("No results to analyze!")
            return

        print("=== UCB-Extract-Order Results Summary ===")
        print(f"Number of instances: {len(results)}")

        slopes = [results[inst_id][1]['slope'] for inst_id in results.keys()]
        slopes = [s for s in slopes if not np.isnan(s)]

        if slopes:
            print(f"Slope statistics:")
            print(f"  Mean: {np.mean(slopes):.3f}")
            print(f"  Std:  {np.std(slopes):.3f}")
            print(f"  Min:  {np.min(slopes):.3f}")
            print(f"  Max:  {np.max(slopes):.3f}")
            print(f"  Reference (√T): 0.500")
            print(f"  Difference from √T: {np.mean(slopes) - 0.5:.3f}")

        print(f"Instance IDs: {sorted(results.keys())}")


def main():
    """Example usage of the plotting script."""
    plotter = UCBEOResultsPlotter()

    # Print summary statistics
    plotter.summary_statistics()

    # Plot mean with standard deviation area
    print("\nPlotting mean with std area...")
    plotter.plot_with_reference(reference_intercept=6.7, show_mean_std=True)


if __name__ == "__main__":
    main()
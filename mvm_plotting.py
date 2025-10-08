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

class MVMResultsPlotter:
    def __init__(self, results_dir="mvm_results", plots_dir="mvm_plots"):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def load_results(self, instance_ids=None):
        """
        Load results from CSV files.

        Args:
            instance_ids: List of instance IDs to load. If None, loads all available.

        Returns:
            Dict mapping instance_id to (dataframe, metadata)
        """
        results = {}

        # Find all CSV files in results directory
        csv_pattern = os.path.join(self.results_dir, 'mvm_regret_results_instance_*.csv')
        csv_files = glob.glob(csv_pattern)

        # Extract instance IDs from filenames
        available_instances = []
        for csv_file in csv_files:
            match = re.search(r'instance_(\d+)\.csv', csv_file)
            if match:
                available_instances.append(int(match.group(1)))

        available_instances.sort()

        # Determine which instances to load
        if instance_ids is None:
            instance_ids = available_instances
        else:
            # Check if requested instances exist
            missing = set(instance_ids) - set(available_instances)
            if missing:
                print(f"Warning: Instances {missing} not found. Available: {available_instances}")
            instance_ids = [i for i in instance_ids if i in available_instances]

        # Load the data
        for inst_id in instance_ids:
            csv_file = os.path.join(self.results_dir, f'mvm_regret_results_instance_{inst_id}.csv')
            try:
                df = pd.read_csv(csv_file)

                # Try to load metadata from CSV attributes (pandas doesn't always preserve this)
                # So we'll extract slope from the fitted line
                if len(df) >= 2:
                    # Calculate slope from fitted_log_regret column
                    log_t = df['log_t'].values
                    fitted_log = df['fitted_log_regret'].values
                    slope = (fitted_log[-1] - fitted_log[0]) / (log_t[-1] - log_t[0])
                else:
                    slope = np.nan

                metadata = {
                    'slope': slope,
                    'instance': inst_id
                }

                results[inst_id] = (df, metadata)

            except Exception as e:
                print(f"Error loading instance {inst_id}: {e}")

        return results

    def plot_mean_with_std(self, instance_ids=None, save_individual=False,
                          show_reference_line=True, reference_slope=0.5,
                          reference_intercept=None, auto_intercept_offset=0.5):
        """
        Plot mean regret curve across all instances with standard deviation area.

        Args:
            instance_ids: List of instance IDs to plot. If None, plots all.
            save_individual: Whether to save individual plots for each instance
            show_reference_line: Whether to show reference line with given slope
            reference_slope: Slope of the reference line (default 0.5 for √T growth)
            reference_intercept: Fixed intercept for reference line. If None, auto-calculate.
            auto_intercept_offset: Offset above max regret when auto-calculating intercept
        """
        results = self.load_results(instance_ids)

        if not results:
            print("No results to plot!")
            return

        # Collect all data and align by time points
        all_log_t = []
        all_regret_data = []

        for inst_id, (df, metadata) in results.items():
            all_log_t.extend(df['log_t'].values)
            all_regret_data.append(df)

        # Find common time range
        min_log_t = max([df['log_t'].min() for df in all_regret_data])
        max_log_t = min([df['log_t'].max() for df in all_regret_data])

        # Create common time grid
        n_points = min([len(df) for df in all_regret_data])  # Use minimum length
        common_log_t = np.linspace(min_log_t, max_log_t, n_points)

        # Interpolate all instances to common time grid
        regret_matrix = np.zeros((len(results), n_points))

        for i, (inst_id, (df, metadata)) in enumerate(results.items()):
            # Interpolate to common grid
            regret_interp = np.interp(common_log_t, df['log_t'].values, df['log_mean_regret'].values)
            regret_matrix[i, :] = regret_interp

        # Calculate mean and standard deviation across instances
        mean_regret = np.mean(regret_matrix, axis=0)
        std_regret = np.std(regret_matrix, axis=0)

        # Create combined plot
        plt.figure(figsize=(12, 8))

        # Plot reference line first (so it appears behind other lines)
        if show_reference_line:
            # Determine intercept
            if reference_intercept is None:
                # Auto-calculate intercept: position above the data
                max_regret = np.max(mean_regret + std_regret)
                ref_intercept = max_regret + auto_intercept_offset
            else:
                ref_intercept = reference_intercept

            ref_log_regret = reference_slope * common_log_t + ref_intercept

            # Create label based on slope
            if reference_slope == 0.5:
                ref_label = r'$\sqrt{T}$ growth'
            else:
                ref_label = f'slope = {reference_slope}'

            plt.plot(common_log_t, ref_log_regret,
                    color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle=':', alpha=0.8,
                    label=ref_label)

        # Plot standard deviation area
        plt.fill_between(common_log_t,
                        mean_regret - std_regret,
                        mean_regret + std_regret,
                        alpha=0.3, color='blue', label='±1 std')

        # Plot mean line
        plt.plot(common_log_t, mean_regret,
                color='blue', linewidth=MAIN_LINE_WIDTH,
                label=f'Mean across {len(results)} instances')

        plt.xlabel('log(round $t$)', fontsize=LABEL_FONTSIZE)
        plt.ylabel('log(cumulative regret)', fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()

        # Save combined plot as PDF
        combined_filename = f"mvm_mean_std_instances_{'_'.join(map(str, results.keys()))}.pdf"
        plt.savefig(os.path.join(self.plots_dir, combined_filename), dpi=300, bbox_inches='tight')

        if save_individual:
            # Save individual plots (keeping original functionality)
            for inst_id, (df, metadata) in results.items():
                plt.figure(figsize=(10, 6))

                # Plot reference line for individual plots too
                if show_reference_line:
                    ref_log_t = np.linspace(df['log_t'].min(), df['log_t'].max(), 100)

                    # Use same intercept logic for individual plots
                    if reference_intercept is None:
                        max_regret_individual = df['log_mean_regret'].max()
                        ref_intercept = max_regret_individual + (auto_intercept_offset * 0.6)
                    else:
                        ref_intercept = reference_intercept

                    ref_log_regret = reference_slope * ref_log_t + ref_intercept

                    # Create label for individual plots
                    if reference_slope == 0.5:
                        ref_label = r'$\sqrt{T}$ growth'
                    else:
                        ref_label = f'slope = {reference_slope}'

                    plt.plot(ref_log_t, ref_log_regret,
                            color='black', linewidth=INDIVIDUAL_LINE_WIDTH, linestyle=':', alpha=0.8,
                            label=ref_label)

                # Plot confidence interval
                if 'log_q5_regret' in df.columns and 'log_q95_regret' in df.columns:
                    plt.fill_between(df['log_t'], df['log_q5_regret'], df['log_q95_regret'],
                                   alpha=0.3, label="5th–95th percentile")

                # Plot mean regret
                plt.plot(df['log_t'], df['log_mean_regret'],
                        linewidth=INDIVIDUAL_LINE_WIDTH, label="Mean Log Regret")

                plt.xlabel('log(round $t$)', fontsize=LABEL_FONTSIZE)
                plt.ylabel('log(cumulative regret)', fontsize=LABEL_FONTSIZE)
                plt.title(f'Instance {inst_id}: Log–Log Regret Growth', fontsize=TITLE_FONTSIZE)
                plt.legend(fontsize=LEGEND_FONTSIZE)
                plt.grid(alpha=0.3)
                plt.tick_params(labelsize=TICK_FONTSIZE)
                plt.tight_layout()

                # Save individual plots as PDF
                plt.savefig(os.path.join(self.plots_dir, f'instance_{inst_id}_regret_plot.pdf'),
                          dpi=300, bbox_inches='tight')
                plt.close()

        plt.show()

    def plot_individual_instances(self, instance_ids=None, show_slopes=False,
                                show_confidence=True, save_individual=True,
                                show_reference_line=True, reference_slope=0.5,
                                reference_intercept=None, auto_intercept_offset=0.5):
        """
        Plot individual log-log regret curves for each instance (original method).
        """
        results = self.load_results(instance_ids)

        if not results:
            print("No results to plot!")
            return

        # Generate colors for different instances
        colors = cm.tab10(np.linspace(0, 1, len(results)))

        # Create combined plot
        plt.figure(figsize=(12, 8))

        # Find the range of log_t values for reference line
        all_log_t = []
        for inst_id, (df, metadata) in results.items():
            all_log_t.extend(df['log_t'].values)

        min_log_t = min(all_log_t)
        max_log_t = max(all_log_t)

        # Plot reference line first (so it appears behind other lines)
        if show_reference_line:
            # Create reference line that spans the data range
            ref_log_t = np.linspace(min_log_t, max_log_t, 100)

            # Determine intercept
            if reference_intercept is None:
                # Auto-calculate intercept: position above the data
                max_regret = max([df['log_mean_regret'].max() for _, (df, _) in results.items()])
                ref_intercept = max_regret + auto_intercept_offset
            else:
                ref_intercept = reference_intercept

            ref_log_regret = reference_slope * ref_log_t + ref_intercept

            # Create label based on slope
            if reference_slope == 0.5:
                ref_label = r'$\sqrt{T}$ growth'
            else:
                ref_label = f'slope = {reference_slope}'

            plt.plot(ref_log_t, ref_log_regret,
                    color='black', linewidth=REFERENCE_LINE_WIDTH, linestyle=':', alpha=0.8,
                    label=ref_label)

        for i, (inst_id, (df, metadata)) in enumerate(results.items()):
            color = colors[i]

            # Plot confidence interval if requested
            if show_confidence and 'log_q5_regret' in df.columns and 'log_q95_regret' in df.columns:
                plt.fill_between(df['log_t'], df['log_q5_regret'], df['log_q95_regret'],
                               alpha=0.2, color=color)

            # Plot mean regret
            line, = plt.plot(df['log_t'], df['log_mean_regret'],
                           color=color, linewidth=MAIN_LINE_WIDTH,
                           label=f'Instance {inst_id}')

        plt.xlabel('log(round $t$)', fontsize=LABEL_FONTSIZE)
        plt.ylabel('log(cumulative regret)', fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()

        # Save combined plot as PDF
        combined_filename = f"mvm_combined_instances_{'_'.join(map(str, results.keys()))}.pdf"
        plt.savefig(os.path.join(self.plots_dir, combined_filename), dpi=300, bbox_inches='tight')

        plt.show()

    def plot_with_reference(self, instance_ids=None, reference_slope=0.5,
                          reference_intercept=None, auto_intercept_offset=0.5,
                          show_mean_std=True):
        """
        Convenience method to plot with reference line.

        Args:
            instance_ids: List of instance IDs to plot
            reference_slope: Slope of reference line (0.5 for √T)
            reference_intercept: Fixed intercept. If None, auto-calculate.
            auto_intercept_offset: Offset above data when auto-calculating
            show_mean_std: If True, show mean with std area. If False, show individual lines.
        """
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
        """
        Print summary statistics of the results.
        """
        results = self.load_results(instance_ids)

        if not results:
            print("No results to analyze!")
            return

        print("=== MVM Results Summary ===")
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
    """
    Example usage of the plotting script.
    """
    plotter = MVMResultsPlotter()

    # Print summary statistics
    plotter.summary_statistics()

    # Plot mean with standard deviation area
    print("\nPlotting mean with std area...")
    plotter.plot_with_reference(reference_intercept=6.7, show_mean_std=True)

    # Optional: Plot individual lines (original behavior)
    # print("\nPlotting individual instances...")
    # plotter.plot_with_reference(reference_intercept=6.7, show_mean_std=False)


if __name__ == "__main__":
    main()
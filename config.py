

# shared parameters
U = 100
K = 20
N_INSTANCES = 10
N_RUNS = 10

# varying parameters
T_MvM = 100_000
T_EMC_LIST = [i * 100_000 for i in range(1, 11)]


# Global font settings - easy to modify
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 18
TICK_FONTSIZE = 16

# Global line width settings - easy to modify
REFERENCE_LINE_WIDTH = 6        # Reference line (√T)
MAIN_LINE_WIDTH = 3            # Main regret curves
INDIVIDUAL_LINE_WIDTH = 3      # Individual plot lines

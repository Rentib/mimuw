import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load the benchmark data
df = pd.read_csv('bench.csv')

# Create a configuration column that combines the optimization flags
def get_configuration(row):
    config = []
    if row['enable_ios'] == 1:
        config.append('I')
    if row['enable_push_pull_pruning'] == 1:
        config.append('P')
    if row['enable_hybridization'] == 1:
        config.append('H')
    if row['enable_load_balancing'] == 1:
        config.append('L')
    return '+'.join(config) if config else 'baseline'

df['configuration'] = df.apply(get_configuration, axis=1)

# Filter for delta=25 and tau=0.4 for the first analysis
df_filtered = df[(df['delta'] == 25) & (df['hybridization_tau'] == 0.4)].copy()

# Calculate mean runtimes for each configuration
mean_runtimes = df_filtered.groupby(['processes', 'vertices', 'configuration'])['elapsed_time'].mean().reset_index()

# Get unique process counts
process_counts = sorted(df_filtered['processes'].unique())

# Create one plot per process count
for processes in process_counts:
    plt.figure(figsize=(10, 6))
    subset = mean_runtimes[mean_runtimes['processes'] == processes]
    
    # Plot each configuration
    for config in subset['configuration'].unique():
        config_data = subset[subset['configuration'] == config]
        plt.plot(config_data['vertices'], config_data['elapsed_time'], 
                marker='o', linestyle='-', linewidth=2,
                label=config)
    
    # Set log scales with custom formatting
    ax = plt.gca()
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    
    # Custom y-axis labels (10, 1, 0.1 instead of 10^1, 10^0, 10^-1)
    yticks = [0.1, 1, 10, 100]
    plt.yticks(yticks, [f"{t:.1f}" if t < 1 else f"{t:.0f}" for t in yticks])
    
    plt.title(f'Runtime vs Problem Size ({processes} processes)', pad=15)
    plt.xlabel('Vertices (log₂ scale)', labelpad=10)
    plt.ylabel('Time [seconds] (log₁₀ scale)', labelpad=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Improved legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., framealpha=1)
    
    plt.tight_layout()
    plt.savefig(f'runtime_vs_problem_size_{processes}_processes.png')

# Find the configuration with lowest mean time for each process count
optimal_configs = []
for processes in process_counts:
    process_data = mean_runtimes[mean_runtimes['processes'] == processes]
    # Get mean time across all problem sizes for each configuration
    config_means = process_data.groupby('configuration')['elapsed_time'].mean()
    optimal_config = config_means.idxmin()
    optimal_configs.append((processes, optimal_config))
    print(f"Optimal for {processes} processes: {optimal_config}")

# For simplicity, we'll take the most frequently optimal configuration as our OPT
# Alternatively, you could choose based on your most common process count
OPT = mode([config for (proc, config) in optimal_configs])
print(f"\nSelected OPT configuration: {OPT}")

# Filter data for our OPT configuration and tau=0.4
df_opt = df[(df['configuration'] == OPT) & (df['hybridization_tau'] == 0.4)].copy()

# Get unique delta values
delta_values = sorted(df_opt['delta'].unique())

# Create one plot per process count
for processes in process_counts:
    plt.figure()
    subset = df_opt[df_opt['processes'] == processes]
    
    # Plot each delta value
    for delta in delta_values:
        delta_data = subset[subset['delta'] == delta]
        mean_times = delta_data.groupby('vertices')['elapsed_time'].mean()
        plt.plot(mean_times.index, mean_times.values, 
                marker='o', label=f'delta={delta}')
    
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.title(f'OPT ({OPT}) Runtime vs Problem Size ({processes} processes)')
    plt.xlabel('Vertices (log scale)')
    plt.ylabel('Time (log scale)')
    yticks = [0.1, 1, 10, 100]
    plt.yticks(yticks, [f"{t:.1f}" if t < 1 else f"{t:.0f}" for t in yticks])
    plt.legend()
    plt.savefig(f'opt_runtime_vs_problem_size_{processes}_processes.png')

# Determine best delta (lowest average time across all problem sizes)
opt_delta_means = df_opt.groupby(['processes', 'delta'])['elapsed_time'].mean().reset_index()
best_deltas = []
for processes in process_counts:
    process_data = opt_delta_means[opt_delta_means['processes'] == processes]
    best_delta = process_data.loc[process_data['elapsed_time'].idxmin(), 'delta']
    best_deltas.append((processes, best_delta))
    print(f"Best delta for {processes} processes: {best_delta}")

# Select most frequently best delta
BEST_DELTA = mode([delta for (proc, delta) in best_deltas])
print(f"\nSelected BEST_DELTA: {BEST_DELTA}")

# Filter data for our OPT configuration and BEST_DELTA
df_opt_tau = df[(df['configuration'] == OPT) & (df['delta'] == BEST_DELTA)].copy()

# Get unique tau values
tau_values = sorted(df_opt_tau['hybridization_tau'].unique())

# Create one plot per process count
for processes in process_counts:
    plt.figure()
    subset = df_opt_tau[df_opt_tau['processes'] == processes]
    
    # Plot each tau value
    for tau in tau_values:
        tau_data = subset[subset['hybridization_tau'] == tau]
        mean_times = tau_data.groupby('vertices')['elapsed_time'].mean()
        plt.plot(mean_times.index, mean_times.values, 
                marker='o', label=f'tau={tau:.2f}')
    
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.title(f'OPT ({OPT}) Runtime vs Problem Size ({processes} processes)\nDelta={20}')
    plt.xlabel('Vertices (log scale)')
    plt.ylabel('Time (log scale)')
    yticks = [0.1, 1, 10, 100]
    plt.yticks(yticks, [f"{t:.1f}" if t < 1 else f"{t:.0f}" for t in yticks])
    plt.legend()
    plt.savefig(f'opt_runtime_vs_problem_size_{processes}_processes_delta_{BEST_DELTA}.png')

# Determine best tau (lowest average time across all problem sizes)
opt_tau_means = df_opt_tau.groupby(['processes', 'hybridization_tau'])['elapsed_time'].mean().reset_index()
best_taus = []
for processes in process_counts:
    process_data = opt_tau_means[opt_tau_means['processes'] == processes]
    best_tau = process_data.loc[process_data['elapsed_time'].idxmin(), 'hybridization_tau']
    best_taus.append((processes, best_tau))
    print(f"Best tau for {processes} processes: {best_tau:.2f}")

# Select most frequently best tau
BEST_TAU = mode([tau for (proc, tau) in best_taus])
print(f"\nSelected BEST_TAU: {BEST_TAU:.2f}")

# Prepare data - filter for best parameters
df_weak = df[(df['delta'] == 25) & (df['hybridization_tau'] == 0.4)].copy()

# Get sorted unique process counts
process_counts = sorted(df_weak['processes'].unique())
min_processes = min(process_counts)
print(f"Process counts available: {process_counts}")

# Get configurations actually present in the data
available_configs = sorted(df_weak['configuration'].unique())
print(f"Available configurations: {available_configs}")

# Create a consistent color mapping with unique colors
color_map = {
    'baseline': 'C0',  # Consistent baseline color
    OPT: 'C1',         # Highlight OPT with distinct color
}
remaining_configs = [c for c in available_configs if c not in color_map]
for i, config in enumerate(remaining_configs, start=2):  # Start at C2
    color_map[config] = f'C{i}'

# Define plotting order (OPT first, then baseline, then others sorted)
plotting_order = [OPT, 'baseline'] + sorted([c for c in available_configs 
                                           if c not in [OPT, 'baseline']])

# Problem sizes to analyze
problem_sizes = sorted(df_weak['vertices'].unique())[::2]  # Show every other size

for size in problem_sizes:
    plt.figure(figsize=(12, 7))
    size_data = df_weak[df_weak['vertices'] == size]
    
    # Plot ideal linear speedup reference
    ideal_speedup = [p/min_processes for p in process_counts]
    plt.plot(process_counts, ideal_speedup, 'k--', label='Ideal Speedup', linewidth=2)
    
    # Calculate efficiencies first
    config_efficiencies = {}
    for config in plotting_order:
        config_data = size_data[size_data['configuration'] == config]
        if len(config_data) == 0:
            continue
            
        # Get baseline time for this config
        config_baseline = config_data[config_data['processes'] == min_processes]['elapsed_time'].mean()
        
        # Calculate speedup and efficiency
        mean_times = config_data.groupby('processes')['elapsed_time'].mean().sort_index()
        speedup = config_baseline / mean_times
        last_speedup = speedup.iloc[-1]
        ideal_last = ideal_speedup[-1]
        efficiency = last_speedup/ideal_last
        config_efficiencies[config] = efficiency
        
        # Plot with assigned color
        plt.plot(mean_times.index, speedup, 
                marker='o', linestyle='-', linewidth=2,
                color=color_map[config])
    
    # Format vertices as power of 2 in title
    n = int(np.log2(size))
    plt.title(f'Weak Scaling: Speedup vs Processes\nProblem Size = 2^{n} ({size} vertices)', pad=20, fontsize=14)
    plt.xlabel('Number of Processes', labelpad=10, fontsize=12)
    plt.ylabel(f'Speedup (Relative to {min_processes} processes)', labelpad=10, fontsize=12)
    plt.xticks(process_counts, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Create legend with efficiency percentages
    present_configs = [config for config in plotting_order 
                      if config in size_data['configuration'].unique()]
    legend_entries = []
    for config in present_configs:
        eff = config_efficiencies.get(config, 0)
        legend_entries.append(f"{config} ({eff:.0%} eff.)")
    
    handles = [plt.Line2D([0], [0], color=color_map[config], lw=3) 
              for config in present_configs]
    handles.append(plt.Line2D([0], [0], color='k', linestyle='--', lw=2))  # For ideal line
    legend_entries.append("Ideal Speedup")
    
    plt.legend(handles, legend_entries,
              loc='upper left', bbox_to_anchor=(1, 1),
              title='Configurations', title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'weak_scaling_speedup_vs_processes_size_{size}.png')

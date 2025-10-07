import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results.csv')

# Calculate speedup (time with 1 host / time with N hosts)
host_counts = sorted(df['hosts'].unique())
host_counts = [h for h in host_counts if h > 1]  # Compare against 1-host baseline

speedup_data = {}
for n_hosts in host_counts:
    merged = pd.merge(
        df[df['hosts'] == 1][['n', 'time']],
        df[df['hosts'] == n_hosts][['n', 'time']],
        on='n',
        suffixes=('_single', f'_{n_hosts}')
    )
    speedup_data[n_hosts] = merged['time_single'] / merged[f'time_{n_hosts}']

# Get unique n values (problem sizes)
n_values = sorted(df['n'].unique())
x_positions = range(len(n_values))  # Evenly spaced positions

# Plot
plt.figure(figsize=(12, 6))

# Plot speedup for each host count
for n_hosts in host_counts:
    plt.plot(
        x_positions, 
        speedup_data[n_hosts],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=8,
        label=f'{n_hosts} hosts'
    )

# Set x-axis ticks to show all n values
plt.xticks(x_positions, n_values, rotation=45)
plt.xlabel('Problem size (n)')
plt.ylabel('Speedup (compared to 1 host)')
plt.title('Speedup vs Single Host for Different Host Counts')

# Set y-axis range from 0 to 8
plt.ylim(0, 8)

# Add ideal speedup reference lines (dotted)
for n_hosts in host_counts:
    plt.axhline(y=n_hosts, color='gray', linestyle=':', alpha=0.3)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig('speedup_plot_fixed_yaxis.png', dpi=300)
plt.show()

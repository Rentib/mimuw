import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark.csv")

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: CompressNaiveSpeedup
plt.subplot(1, 3, 1)
for accuracy in df['Accuracy'].unique():
    subset = df[df['Accuracy'] == accuracy]
    plt.plot(subset['Threads'], subset['CompressNaiveSpeedup'], label=f'Accuracy {accuracy}')
plt.xlabel('Number of Threads')
plt.ylabel('Compress Naive Speedup')
plt.title('Compress Naive Speedup vs Threads')
plt.legend()
plt.grid()

# Plot 2: CompressSpeedup
plt.subplot(1, 3, 2)
for accuracy in df['Accuracy'].unique():
    subset = df[df['Accuracy'] == accuracy]
    plt.plot(subset['Threads'], subset['CompressSpeedup'], label=f'Accuracy {accuracy}')
plt.xlabel('Number of Threads')
plt.ylabel('Compress Speedup')
plt.title('Compress Speedup vs Threads')
plt.legend()
plt.grid()

# Plot 3: DecompressSpeedup
plt.subplot(1, 3, 3)
for accuracy in df['Accuracy'].unique():
    subset = df[df['Accuracy'] == accuracy]
    plt.plot(subset['Threads'], subset['DecompressSpeedup'], label=f'Accuracy {accuracy}')
plt.xlabel('Number of Threads')
plt.ylabel('Decompress Speedup')
plt.title('Decompress Speedup vs Threads')
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()

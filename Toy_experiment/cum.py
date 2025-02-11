import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('classical_subset_simulation_N_1600.npy')

# Calculate the cumulative sum
cumulative_data = np.cumsum(data)

# Plot the cumulative data
x = np.linspace(min(data), max(data), len(data))
plt.plot(x, cumulative_data)
plt.title('Cumulative Plot')
plt.xlabel('Index')
plt.ylabel('Cumulative Sum')
plt.grid(True)

# Display the plot
plt.show()

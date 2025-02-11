# To find the burn-in length of the MCMC chain

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
# Initialization
theta = np.random.normal(0, 1, 100)
kappa = np.random.uniform(-1, 1, 100)
G = theta + kappa * 0.5

c_1 = np.sort(G)[10]
print("c_1: ", c_1)

# For l = 2, no burn-in
mask = G <= c_1
theta = theta[mask][:1]
G = G[mask][:10]

for theta_i in theta:
    seed = theta_i
    plot_new_theta = [seed]
    for j in range(1000):
        theta_new = 0.8 * seed + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        G_new = theta_new + kappa * 0.5 ** 2
        
        if G_new <= c_1:
            theta = np.append(theta, theta_new)
            seed = theta_new
            plot_new_theta.append(theta_new)
        else:
            theta = np.append(theta, seed)
            plot_new_theta.append(seed)
            
    plt.plot(plot_new_theta, '.-', label = 'l=2, theta_i')
    
plt.legend()
plt.show()
# Problem setting:
# G ~ N(0, 1)
# G_l(\omega) = G(\omega) + \kapa(\omega) * \gamma^{l} with \kapa(\omega) ~ N(0, 1) and \gamma = 0.5
# q = 2 for the expected costs
# Prob(G <= -3.8) \approx 7.23e-05

# This script is used to simulate the classical subset selection algorithm with full-refinement

import numpy as np

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    n = len(data)
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, n), replace=True)
    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
    
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    # Swap percentiles for wider 99% confidence interval
    lower_percentile_99 = (1.0 - 0.99) / 2.0 * 100
    upper_percentile_99 = (1.0 + 0.99) / 2.0 * 100
    
    lower_bound_95 = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound_95 = np.percentile(bootstrap_estimates, upper_percentile)
    
    lower_bound_99 = np.percentile(bootstrap_estimates, lower_percentile_99)  # use upper_percentile_99 as lower bound
    upper_bound_99 = np.percentile(bootstrap_estimates, upper_percentile_99)  # use lower_percentile_99 as upper bound
    
    return (lower_bound_95, upper_bound_95), (lower_bound_99, upper_bound_99)

import matplotlib.pyplot as plt

from Generate_G_l import *

def classical_subset_simulation(N, y_L = -3.8, p0 = 0.1, gamma = 0.5, L = 5):
    # input:
    # N: total number of samples per level
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    
    # Initialization, l= 1
    G = np.random.normal(0, 1, N)
    kappa = np.array([k(g) for g in G])
    G_l = G + kappa * gamma
    
    # Compute the probability threshold
    c_l = sorted(G_l)[int(N*p0)]
    
    if c_l <= y_L:
        mask = G_l <= y_L
        return np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask]
    
    for l in range(2, L):
        # G_l = Generate_G_l(G_l, N, l, c_l, gamma = gamma)
        # G_l = modified_metropolis_hastings(G_l, N, l, c_l, gamma = gamma)
        G_l = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        
        c_l = sorted(G_l)[int(N*p0)]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            return p0 ** (l-1) * np.sum(mask) / N
        
        mask = G_l <= c_l
        G_l = G_l[mask]
        
    # G_l= Generate_G_l(G_l, N, L, c_l, gamma = gamma)
    # G_l = modified_metropolis_hastings(G_l, N, l, c_l, gamma = gamma)
    G_l = sample_new_G(G_l, N, l, c_l, gamma = gamma)
    mask = G_l <= y_L
    # print("The number of samples in the failure domain is", np.sum(mask))
    return p0 ** (L-1) * np.sum(mask) / N

if __name__ == "__main__":
    N = 1000  # Total number of samples per level
    p_0 = 0.1  # Probability threshold for each subset, 0.15^5 = 7.59e-05
    gamma = 0.5
    L = 5  # Total number of levels
    y_L = -3.8  # Failure threshold
    
    np.random.seed(0)
    # p_f = classical_subset_simulation(N, p0=p_0, L=L)
    # print("The failure p_fability is {:.2e}".format(p_f))
    
    failure_probabilities = [classical_subset_simulation(N, p0=p_0, L=L) for _ in range(1000)]
    # print("Failure probabilities:", failure_probabilities[0:10])

    # Calculate 95% confidence interval using bootstrap method
    confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000, confidence_level=0.95)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    print("99% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    # plt.xlim(1e-5, 1e-3)
    plt.step(p_f, cdf, where='post')
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='99% Confidence Interval')
    plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()
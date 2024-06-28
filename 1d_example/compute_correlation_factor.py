# Compute the autocorrelation function for a high-dimensional Markov Chain.

import numpy as np
import matplotlib.pyplot as plt

def compute_autocorrelation_multidimensional(chain, max_lag = 10):
    # input:
    # chain: a list of the Markov Chain values
    # max_lag: the maximum lag to compute the autocorrelation for
    
    # output:
    # avg_autocorrelation: the average autocorrelation values for lags 0 to max_lag across all dimensions
    
    n_samples = len(chain)
    n_dimensions = chain[0].shape[0]
    chain_array = np.array(chain)
    autocorrelations = np.zeros((max_lag + 1, n_dimensions))

    for dim in range(n_dimensions):
        # print("dim:", dim)
        mean = np.mean(chain_array[:, dim])
        var = np.var(chain_array[:, dim])
        
        for lag in range(max_lag + 1):
            if lag < n_samples:
                cov = np.mean((chain_array[:n_samples - lag, dim] - mean) * (chain_array[lag:, dim] - mean))
                # print("cov:", cov)
            else:
                cov = 0
            autocorrelations[lag, dim] = cov / var
    
    avg_autocorrelation = np.mean(autocorrelations, axis=1)
    return avg_autocorrelation

from kl_expansion import kl_expan
from IoQ import IoQ
from mh_sampling import sampling_theta_list
from scipy.stats import pearsonr

# Example usage
np.random.seed(42)
# Generate a high-dimensional MC example, here a random walk in 150 dimensions
theta_ls = [np.random.normal(0, 1, 150)]
G = [0.535 - IoQ(kl_expan(theta_ls[0]), 100)]
G, theta_ls = sampling_theta_list(10, G, theta_ls, 2, 0.535, 100, 0.8)
# print(theta_ls)

avg_autocorrelation = compute_autocorrelation_multidimensional(theta_ls, 20)
print(avg_autocorrelation)
# expected output: 0.8
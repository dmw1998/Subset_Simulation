# Multilevel estimator for the toy experiment

import numpy as np
import matplotlib.pyplot as plt

from Generate_G_l import *

def mle(N, L_b, y_L = -3.8, p0 = 0.1, gamma = 0.5, L = 5):
    # input:
    # N: total number of samples per level
    # L_b: burn-in length
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    N0 = int(N * p0)
    
    # Initialization, l= 1
    G = np.random.normal(0, 1, N)
    kappa = np.array([k(g) for g in G])
    G_l = G + kappa * gamma
    
    # Compute the probability threshold
    c_l = sorted(G_l)[int(N0)]
    # print("The probability threshold for level 1 is", c_l)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        # print("left at level 1")
        return np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask]
    
    dinominator = 1
    
    # level 2, no burn-in
    G_l = sample_new_G(G_l, N, 1, c_l, gamma = gamma)
    c_l_1 = c_l
    
    c_l = sorted(G_l)[int(N0)]
    # print("The probability threshold for level 2 is", c_l)
    
    if c_l <= y_L:
        # print("left at level 2")
        mask = G_l <= y_L
        return p0 * np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask]
    
    G_l_1 = sample_new_G(G_l, N, 1, c_l, gamma = gamma)
    mask = G_l_1 <= c_l_1
    dinominator *= np.mean(mask)
    # print("The denominator for level 2 is", dinominator)
    
    # level > 2, with burn-in
    N = N + (L_b - 1) * N0
    for l in range(3, L):
        G_l = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        # drop the first L_b samples in each Markov chain
        G_l = G_l[int(N0*L_b):]
        c_l_1 = c_l
        
        c_l = sorted(G_l)[int(N0)]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            # print("left at level", l)
            return p0 ** (l-1) * np.sum(mask) / N / dinominator

        mask = G_l <= c_l
        G_l = G_l[mask]
        
        G_l_1 = sample_new_G(G_l, N, l-1, c_l, gamma = gamma)
        # drop the first L_b samples in each Markov chain
        G_l_1 = G_l_1[int(N0*L_b):]
        mask = G_l_1 <= c_l_1
        dinominator *= np.mean(mask)
        # print("The denominator for level", l, "is", dinominator)
        
    G_l = sample_new_G(G_l, N, L, c_l, gamma = gamma)
    # drop the first L_b samples in each Markov chain
    G_l = G_l[int(N0*L_b):]
    mask = G_l <= y_L
    # print("reach the last level")
    
    return p0 ** (L-1) * np.sum(mask) / N / dinominator

if __name__ == "__main__":
    N = 10000  # Total number of samples per level
    p_0 = 0.1  # Probability threshold for each subset
    gamma = 0.5
    L = 5  # Total number of levels
    y_L = -3.8  # Failure threshold
    
    np.random.seed(0)
    p_f = mle(N, 0, p0=p_0, L=L)
    print("The failure probability is {:.2e}".format(p_f))
    
    from confidence_interval import bootstrap_confidence_interval

    # failure_probabilities = [mle(N, 10, p0=p_0, L=L) for _ in range(1000)]
    # # print("Failure probabilities:", failure_probabilities[0:10])

    # # Calculate 95% confidence interval using bootstrap method
    # confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    # print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    # print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    # p_f = sorted(failure_probabilities)
    # cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # # Plot the empirical CDF
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
    # # plt.xlim(1e-5, 1e-3)
    # plt.step(p_f, cdf, where='post')
    # plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    # plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    # plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    # plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='90% Confidence Interval')
    # plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    # plt.xlabel('Probability')
    # plt.ylabel('Empirical CDF')
    # plt.title('Empirical CDF of Probabilities')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    L_b = [0, 5]#, 10, 20]
    p_0 = [0.05, 0.1]#, 0.2, 0.25]
    
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    
    for b in L_b:
        for p in p_0:
            print("")
            print("L_b = {}, p_0 = {}".format(b, p))
            failure_probabilities = [mle(N, b, p0=p, L=L) for _ in range(1000)]
            p_f = sorted(failure_probabilities)
            
            # Calculate 95% confidence interval using bootstrap method
            confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

            print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
            
            print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
            
            cdf = np.arange(1, len(p_f) + 1) / len(p_f)
            plt.step(p_f, cdf, alpha = 0.5, where='post', label='L_b = {}, p_0 = {}'.format(b, p))
            
    from classical_subset_simulation import classical_subset_simulation
            
    N = 1000  # Total number of samples per level
    p_0 = 0.1  # Probability threshold for each subset
    gamma = 0.5
    L = 5  # Total number of levels
    y_L = -3.8  # Failure threshold

    failure_probabilities = [classical_subset_simulation(N, p0=p_0, L=L) for _ in range(1000)]
    
    # Calculate 95% confidence interval using bootstrap method
    confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 
    plt.step(p_f, cdf, color = 'r', where='post', label='Classical Subset Simulation')

    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()
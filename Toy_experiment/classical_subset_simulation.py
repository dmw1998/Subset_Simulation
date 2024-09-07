# Problem setting:
# G ~ N(0, 1)
# G_l(\omega) = G(\omega) + \kapa(\omega) * \gamma^{l} with \kapa(\omega) ~ N(0, 1) and \gamma = 0.5
# q = 2 for the expected costs
# Prob(G <= -3.8) \approx 7.23e-05

# This script is used to simulate the classical subset selection algorithm with full-refinement

import numpy as np
import matplotlib.pyplot as plt

def cost(N, p0, L):
    c = 2*N
    for l in range(1, L):
        c += (10 + l) * N * p0
    
    return c

def sample_new_G(G_l, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    
    # output:
    # G_l: new samples for next level
    
    N0 = len(G_l)
    
    for i in range(N - N0):
        # Propose a new sample for G ~ N(0,1)
        # Computational cost: 7
        G_new = 0.4 * G_l[i] + np.sqrt(1 - 0.4 ** 2) * np.random.normal(0, 1)
        # Add noise
        kappa_new = np.random.uniform(-1, 1)
        # Compute the new G_l
        # Computational cost: 2 + l
        G_l_new = G_new + kappa_new * gamma ** l        
        # Computational cost: 1
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
        else:
            G_l = np.append(G_l, G_l[i])
            
    # Computational cost in total: (10 + l) * N0
            
    return G_l

def classical_subset_simulation(N, y_L = -3.8, p0 = 0.1, gamma = 0.5, L = 5):
    # input:
    # N: total number of samples per level
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    N0 = int(N * p0)
    
    # Initialization, l= 1
    # Computational cost: 2N
    G = np.random.normal(0, 1, N)
    kappa = np.array([np.random.uniform(-1, 1) for _ in G])
    G_l = G + kappa * gamma
    
    # Compute the probability threshold
    c_l = sorted(G_l)[N0-1]
    # print("The probability threshold for level 1 is", c_l)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        return np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    for l in range(2, L):
        G_l = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        
        c_l = sorted(G_l)[N0-1]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            return p0 ** (l-1) * np.sum(mask) / N
        
        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        
    G_l = sample_new_G(G_l, N, L, c_l, gamma = gamma)
    mask = G_l <= y_L
    # print("The number of samples in the failure domain is", np.sum(mask))
    return p0 ** (L-1) * np.sum(mask) / N

if __name__ == "__main__":
    # N = 1200  # Total number of samples per level
    p_0 = 0.2  # Probability threshold for each subset
    gamma = 0.5
    L = 6  # Total number of levels
    y_L = -3.8  # Failure threshold
    
    cost_list = []
    err_list = []
    for N in [100, 400, 800, 1000, 1600                                              ]:
        np.random.seed(0)
        print("Number of samples per level:", N)
        failure_probabilities = []
        for i in range(100):
            p_f = classical_subset_simulation(N, p0=p_0, L=L)
            # print("The failure p_fability is {:.2e}".format(p_f))
            failure_probabilities.append(p_f)
            
        c = cost(N, p0=p_0, L=L)
        cost_list.append(c)
        p = np.mean(failure_probabilities)
        err_list.append(np.abs(p - 7.23e-05) / 7.23e-05)
        
        print("Failure probabilities:{:.2e}".format(p))
        
    print("The average cost is", cost_list)
    
    
    # Define the principle line
    x = np.linspace(0.01, 0.5, 100)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(err_list, cost_list, marker='o')
    plt.loglog(x, 500 * x ** (-1), 'r--',  label=r'O($\epsilon^{-2}$)')
    plt.xlabel('Relative Error')
    plt.ylabel('Cost')
    plt.title('Classical Subset Simulation')
    plt.show()
    
    # from confidence_interval import bootstrap_confidence_interval

    # failure_probabilities = [classical_subset_simulation(N, p0=p_0, L=L) for _ in range(100)]
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
    
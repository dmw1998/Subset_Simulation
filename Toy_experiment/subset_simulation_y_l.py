import numpy as np
import time
import matplotlib.pyplot as plt

from adaptive_multilevel_subset_simulation import rRMSE

def k(w):
    # return 0.5 if -1 <= w <= 1 else 0
    # return np.random.choice([-1, 1])
    return np.random.uniform(-1, 1) / 3
    # return 0.1

def sample_new_G(G_l, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    
    # output:
    # G_l: new samples for next level
    
    N0 = len(G_l)
    
    if N0 == 0:
        G_l = np.random.normal(0, 1, N)
        N0 = 1
    
    for i in range(N - N0):
        # Propose a new sample for G ~ N(0,1)
        G_new = 0.8 * G_l[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        # Propose a new noise for kappa ~ U({-1, 1})
        kappa_new = k(G_new)
        # Compute the new G_l
        G_l_new = G_new + kappa_new * gamma ** l
        
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
        else:
            G_l = np.append(G_l, G_l[i])
            
    return G_l

def adaptive_subset_simulation_sr(L, gamma, y_L, N):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    # N: the number of samples per level
    
    # output:
    # p_f: the probability of failure
    # total_cost: the total number of samples used
    
    # To compute the sequence of failure thresholds y_l
    y = [-1.3, -2, -2.8, -3.3, y_L]
    # y = y_l(gamma, y_L, L)
    
    total_cost = 0
    
    # To generate the samples
    while True:
        G = np.random.normal(0, 1, N)
        kappa = np.array([k(g) for g in G])
        G_l = G + kappa * gamma
        total_cost += N
        
        mask = G_l <= y[0]
        
        if mask.sum() > 0:
            p_f = mask.mean()
            G_l = G_l[mask]
            break
        
    for l in range(1, L):
        while True:
            total_cost += N - len(G_l)
            G_l = sample_new_G(G_l, N, l + 1, y[l - 1], gamma)
            
            mask = G_l <= y[l]
            G_l = G_l[mask]
            
            if mask.sum() > 0:
                p_f *= mask.mean()
                break
        
    return p_f, total_cost

if __name__ == "__main__":
    L = 5
    gamma = 0.1
    y_L = -3.8
    N = 1000
    TOL = 0.03
    
    np.random.seed(1)
    num_simulations = 1000
    results = [adaptive_subset_simulation_sr(L, gamma, y_L, N) for _ in range(num_simulations)]
    failure_probabilities, costs = zip(*results)
    
    mean_failure_probability = np.mean(failure_probabilities)
    print("The mean of the failure probability: ", mean_failure_probability)
    
    print("The mean of the total cost: ", np.mean(costs))
    
    from confidence_interval import bootstrap_confidence_interval
    
    # Calculate 95% confidence interval using bootstrap method
    confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    p_f_sorted = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f_sorted) + 1) / len(p_f_sorted)

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.step(p_f_sorted, cdf, where='post')
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    plt.axvline(confidence_interval[0], color='g', alpha=0.5, linestyle='--', label='95% Confidence Interval')
    plt.axvline(confidence_interval[1], color='g', alpha=0.5, linestyle='--')
    plt.axvline(ci[0], color='m', alpha=0.5, linestyle='--', label='90% Confidence Interval')
    plt.axvline(ci[1], color='m', alpha=0.5, linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    # Calculate relative errors
    # relative_errors = [rRMSE(p, N) for p, N in zip(failure_probabilities, costs)]
    
    # Plot the distribution of costs
    plt.figure(figsize=(8, 6))
    plt.hist(costs, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Distribution of Costs')
    plt.grid(True)
    plt.show()

import numpy as np
import time
import matplotlib.pyplot as plt

# from adaptive_multilevel_subset_simulation import rRMSE

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    
    difference = p_hat - 7.23e-05
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 7.23e-05

def sample_new_G(G_l, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    
    # output:
    # G_l: new samples for next level
    
    cost = 0
    N0 = len(G_l)
    
    for i in range(N - N0):
        w_new = 0.8 * G_l[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        add_term = kappa * gamma
        tol = gamma
        G_new = w_new + add_term
        cost += gamma ** (-2)
        for j in range(2, l):
            if tol >= np.abs(G_new - c_l):
                tol *= gamma
                add_term *= gamma
                G_new = w_new + add_term
                cost += gamma ** (-2 * j)
            else:
                break
            
        if G_new <= c_l:
            G_l = np.append(G_l, G_new)
        else:
            G_l = np.append(G_l, G_l[i])
        cost += 1
            
    return G_l, cost

def adaptive_subset_simulation_sr(L, gamma, y_L, N):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    # N: the number of samples per level
    
    # output:
    # p_f: the probability of failure
    # total_cost: the total number of samples used
    
    p0 = 0.2
    N0 = int(N * p0)
    
    # Initialization, l= 1
    # Computational cost: 2N
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma
    
    cost = N * gamma ** (-2)
    
    # Compute the probability threshold
    c_l = sorted(G_l)[N0-1]
    # print("The probability threshold for level 1 is", c_l)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        return np.mean(mask), cost
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    for l in range(2, L):
        G_l, add_cost = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        cost += add_cost
        
        c_l = sorted(G_l)[N0-1]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            return p0 ** (l-1) * np.mean(mask), cost
        
        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        
    G_l, add_cost = sample_new_G(G_l, N, L, c_l, gamma = gamma)
    cost += add_cost
    
    mask = G_l <= y_L
    # print("The number of samples in the failure domain is", np.sum(mask))
    
    return p0 ** (L-1) * np.mean(mask), cost
        

if __name__ == "__main__":
    L = 6
    gamma = 0.5
    y_L = -3.8
    cost_list = []
    err_list = []
    
    np.random.seed(2)
    for N in [1000, 5000, 10000, 100000]:
        print("Number of samples per level:", N)
        # np.random.seed(0)
        failure_probabilities = []
        costs = []
        for i in range(100):
            # print("Simulation: ", i)
            p_f, c = adaptive_subset_simulation_sr(L, gamma, y_L, N)
            failure_probabilities.append(p_f)
            costs.append(c)
        
        mean_failure_probability = np.mean(failure_probabilities)
        print("The mean of the failure probability: {:.2e}".format(mean_failure_probability))
        
        cost = np.mean(costs)
        print("The mean of the total cost: {:.2e}".format(cost))
        cost_list.append(cost)
        
        err = rRMSE(failure_probabilities)
        print("The relative error is: {:.2e}\n".format(err))
        err_list.append(err)
        
    # x = np.linspace(2e-2, 1, 100)
    # plt.figure(figsize=(8, 6))
    # plt.loglog(err_list, cost_list, marker='o')
    # plt.loglog(x, 25000 * x ** (-1/2), 'r--',  label=r'O($\epsilon^{-1/2}$)')
    # plt.xlabel('Relative Error')
    # plt.ylabel('Cost')
    # plt.title('Adaptive Subset Simulation')
    # plt.legend()
    # plt.show()
    
    # from confidence_interval import bootstrap_confidence_interval
    
    # # Calculate 95% confidence interval using bootstrap method
    # confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    # print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    # print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    # p_f_sorted = sorted(failure_probabilities)
    # cdf = np.arange(1, len(p_f_sorted) + 1) / len(p_f_sorted)

    # # Plot the empirical CDF
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
    # plt.step(p_f_sorted, cdf, where='post')
    # plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    # plt.axvline(confidence_interval[0], color='g', alpha=0.5, linestyle='--', label='95% Confidence Interval')
    # plt.axvline(confidence_interval[1], color='g', alpha=0.5, linestyle='--')
    # plt.axvline(ci[0], color='m', alpha=0.5, linestyle='--', label='90% Confidence Interval')
    # plt.axvline(ci[1], color='m', alpha=0.5, linestyle='--')
    # plt.xlabel('Probability')
    # plt.ylabel('Empirical CDF')
    # plt.title('Empirical CDF of Probabilities')
    # plt.legend()
    # plt.grid(True)
    # # plt.show()
    
    # # Calculate relative errors
    # # relative_errors = [rRMSE(p, N) for p, N in zip(failure_probabilities, costs)]
    
    # # Plot the distribution of costs
    # plt.figure(figsize=(8, 6))
    # plt.hist(costs, bins=50, edgecolor='black', alpha=0.7)
    # plt.xlabel('Cost')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Costs')
    # plt.grid(True)
    # plt.show()

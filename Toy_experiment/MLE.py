# Multilevel estimator for the toy experiment

import numpy as np
import matplotlib.pyplot as plt

def sample_new_G(G_l, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    # l: current level
    # c_l: probability threshold for each subset
    # gamma: auto-correlation factor
    
    # output:
    # G_l: new samples for next level
    # kappa: new samples for next level
    
    N0 = len(G_l)
    
    for i in range(N - N0):
        G_l_new = 0.8 * G_l[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa_new = np.random.uniform(-1, 1)
        G_l_new += kappa_new * (gamma ** l)
        
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
        else:
            G_l = np.append(G_l, G_l[i])
        
    return G_l


def mle(N, L_b, y_L = -3.8, p0 = 0.1, gamma = 0.5, L = 5):
    # input:
    # N: total number of samples per level
    # L_b: burn-in length
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    N0 = int(N * p0)
    cost = 0
    
    # Initialization, l= 1
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma
    cost += 2 * N
    
    # Compute the probability threshold
    c_l = sorted(G_l)[int(N0)-1]
    # print("The probability threshold for level 1 is", c_l)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        # print("left at level 1")
        return np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    dinominator = 1
    
    # level 2, no burn-in
    G_l = sample_new_G(G_l, N, 2, c_l, gamma = gamma)
    cost += 10 * (N - N0)
    c_l_1 = c_l
    
    c_l = sorted(G_l)[int(N0)-1]
    # print("The probability threshold for level 2 is", c_l)
    
    if c_l <= y_L:
        # print("left at level 2")
        mask = G_l <= y_L
        return p0 * np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    G_l_1 = sample_new_G(G_l, N, 2, c_l, gamma = gamma)
    cost += 10 * (N - N0)
    mask = G_l_1 <= c_l_1
    dinominator *= np.mean(mask)
    # print("The denominator for level 2 is", dinominator)
    
    # level > 2, with burn-in
    N += L_b * N0
    for l in range(3, L):
        G_l = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        cost += (8 + l) * (N - N0)
        # drop the first L_b samples in each Markov chain
        G_l = G_l[int(N0*L_b):]
        c_l_1 = c_l
        
        c_l = sorted(G_l)[int(N0)]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            # print("left at level", l)
            return p0 ** (l-1) * np.sum(mask) / N / dinominator, cost

        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        
        G_l_1 = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        cost += (8 + l) * (N - N0)
        # drop the first L_b samples in each Markov chain
        G_l_1 = G_l_1[int(N0*L_b):]
        mask = G_l_1 <= c_l_1
        dinominator *= np.mean(mask)
        # print("The denominator for level", l, "is", dinominator)
        
    G_l = sample_new_G(G_l, N, L, c_l, gamma = gamma)
    cost += (8 + L) * (N - N0)
    # drop the first L_b samples in each Markov chain
    G_l = G_l[int(N0*L_b):]
    mask = G_l <= y_L
    # print("reach the last level")
    
    return p0 ** (L-1) * np.sum(mask) / N / dinominator, cost

if __name__ == "__main__":
    N = 1000  # Total number of samples per level
    p_0 = 0.25  # Probability threshold for each subset
    gamma = 0.5
    L = 7  # Total number of levels
    y = -3.8  # Failure threshold
    err_list = []
    cost_list = []
    
    # np.random.seed(0)
    for N in [100, 1000, 1600, 3500, 4000]:
    # for N in [4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        print("N = ", N)
        np.random.seed(0)
        failure_probabilities = []
        cost = []
        for i in range(100):
            p_f, c = mle(N, 1, y_L=y, p0=p_0, L=L)
            failure_probabilities.append(p_f)
            cost.append(c)
            
        p = np.mean(failure_probabilities)
        print("The failure probability is {:.2e}".format(p))
        err = abs(p - 7.23e-05) / 7.23e-05
        print("The relative error is {:.2e}".format(err))
        err_list.append(err)
        cost_list.append(np.mean(cost))
        
    x = np.linspace(3e-2, 3e-1, 100)
    plt.figure(figsize=(8, 6))
    plt.loglog(err_list, cost_list, 'o-')
    plt.loglog(x, 530 * x ** (-2), 'r--',  label=r'O($\epsilon^{-2}$)')
    plt.xlabel('Relative Error')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()
    
    # from confidence_interval import bootstrap_confidence_interval

    # # failure_probabilities = [mle(N, 10, p0=p_0, L=L) for _ in range(1000)]
    # # # print("Failure probabilities:", failure_probabilities[0:10])

    # # # Calculate 95% confidence interval using bootstrap method
    # # confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    # # print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    # # print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    # # p_f = sorted(failure_probabilities)
    # # cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # # # Plot the empirical CDF
    # # plt.figure(figsize=(8, 6))
    # # plt.xscale("log")
    # # # plt.xlim(1e-5, 1e-3)
    # # plt.step(p_f, cdf, where='post')
    # # plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    # # plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    # # plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    # # plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='90% Confidence Interval')
    # # plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    # # plt.xlabel('Probability')
    # # plt.ylabel('Empirical CDF')
    # # plt.title('Empirical CDF of Probabilities')
    # # plt.legend()
    # # plt.grid(True)
    # # plt.show()
    
    # L_b = [0, 5]#, 10, 20]
    # p_0 = [0.05, 0.1]#, 0.2, 0.25]
    
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
    # plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    
    # for b in L_b:
    #     for p in p_0:
    #         print("")
    #         print("L_b = {}, p_0 = {}".format(b, p))
    #         failure_probabilities = [mle(N, b, p0=p, L=L) for _ in range(1000)]
    #         p_f = sorted(failure_probabilities)
            
    #         # Calculate 95% confidence interval using bootstrap method
    #         confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    #         print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
            
    #         print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
            
    #         cdf = np.arange(1, len(p_f) + 1) / len(p_f)
    #         plt.step(p_f, cdf, alpha = 0.5, where='post', label='L_b = {}, p_0 = {}'.format(b, p))
            
    # from classical_subset_simulation import classical_subset_simulation
            
    # N = 1000  # Total number of samples per level
    # p_0 = 0.1  # Probability threshold for each subset
    # gamma = 0.5
    # L = 5  # Total number of levels
    # y_L = -3.8  # Failure threshold

    # failure_probabilities = [classical_subset_simulation(N, p0=p_0, L=L) for _ in range(1000)]
    
    # # Calculate 95% confidence interval using bootstrap method
    # confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    # print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    # print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    # p_f = sorted(failure_probabilities)
    # cdf = np.arange(1, len(p_f) + 1) / len(p_f) 
    # plt.step(p_f, cdf, color = 'r', where='post', label='Classical Subset Simulation')

    # plt.xlabel('Probability')
    # plt.ylabel('Empirical CDF')
    # plt.title('Empirical CDF of Probabilities')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
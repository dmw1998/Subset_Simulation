# Solve the toy experiment using the subset simulation method with the selective refinement.

import numpy as np
import time
import matplotlib.pyplot as plt

def subset_simulation_sr(L, gamma, y_L, N):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    # N: the number of samples per level
    
    # output:
    # p_f: the probability of failure
    # p_f_hat: the probability of failure estimated by the subset simulation
    # p_f_hat_sr: the probability of failure estimated by the subset simulation with the selective refinement
    
    # To compute the sequence of failure thresbolds y_l
    # from genrate_y_l import y_l
    y = [-1.3, -2, -2.8, -3.3, y_L]
    cost = np.zeros(L)
    
    # To generate the samples
    while True:
        G = np.random.normal(0, 1, N)
        kappa = np.random.uniform(-1, 1, N)
        G_l = G + kappa * gamma
        cost[0] = 2 * N
        
        mask = G_l <= y[0]
        # print(mask.sum())
        
        if mask.sum() > 0:
            p_f = mask.mean()
            G_l = G_l[mask][:1]
            break
    
    for l in range(2, L+1):
        while True:
            # print("Level: ", l)
            # print("G_l: ", G_l)
            # time.sleep(0.5)
            for i in range(N-1):
                G_l_new = 0.4 * G_l[i] + np.sqrt(1 - 0.4 ** 2) * np.random.normal(0, 1)
                kappa_new = np.random.uniform(-1, 1)
                G_l_new += kappa_new * (gamma ** l)
                cost[l-1] += 8 + l
                
                if G_l_new <= y[l-2]:
                    G_l = np.append(G_l, G_l_new)
                else:
                    G_l = np.append(G_l, G_l[i])
            
            mask = G_l <= y[l-1]
            
            if mask.sum() > 0:
                p_f *= mask.mean()
                G_l = G_l[mask][:1]
                break
            else:
                G_l = G_l[-1:]
                
    return p_f, cost

if __name__ == "__main__":
    L = 5
    gamma = 0.5
    y_L = -3.8
    cost_list = []
    err_list = []
    
    for N in [1000, 1500, 3000]:
        np.random.seed(0)
        failure_probabilities = []
        cost = []
        for _ in range(100):
            p_f, c = subset_simulation_sr(L, gamma, y_L, N)
            failure_probabilities.append(p_f)
            cost.append(c)
            
        ave = np.mean(failure_probabilities)
        print("The mean of the failure probability for N = {}: {:.2e}".format(N, ave))
        err = np.abs(ave - 7.23e-05) / 7.23e-05
        err_list.append(err)
        print("The relative error: {:.2e}".format(err))
        c = np.mean(cost)
        cost_list.append(c)
        print("The average cost: {}".format(c))
        
    x = np.linspace(3e-2, 4e-1, 100)
    plt.figure(figsize=(8, 6))
    plt.loglog(err_list, cost_list, marker='o')
    plt.loglog(x, 5300 * x ** (-1/2), 'r--',  label=r'O($\epsilon^{-1/2}$)')
    plt.xlabel('Relative Error')
    plt.ylabel('Cost')
    plt.title('Subset Simulation with Selective Refinement')
    plt.legend()
    plt.show()
    
    # np.random.seed(1)
    # start = time.time()
    # failure_probabilities = [subset_simulation_sr(L, gamma, y_L, N) for _ in range(100)]
    # print("Time: ", time.time() - start)
    # print("The mean of the failure probability: ", np.mean(failure_probabilities))
    
    # from confidence_interval import bootstrap_confidence_interval
    
    # # Calculate 95% confidence interval using bootstrap method
    # confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    # print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    # print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    # p_f = sorted(failure_probabilities)
    # cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # # Plot the empirical CDF
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
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

    # Time:  23.22992467880249
    # The mean of the failure probability:  7.2273091730631e-05
    # 95% confidence interval for failure probability: (7.07e-05, 7.36e-05)
    # 90% confidence interval for failure probability: (7.10e-05, 7.35e-05)
# Solve the toy experiment using the adaptive multilevel subset simulation method.

import numpy as np
import time
import matplotlib.pyplot as plt

from genrate_y_l import y_l
from Generate_G_l import *

def rRMSE(p_hat, N):
    # input:
    # p_hat: the estimated probability of failure
    # N: the number of samples
    
    # output:
    # rRMSE: the relative root mean square error
    
    # auto-correlation factor \phi = 0.8
    # delta^{2} = \frac{1 - \hat{p}}{\hat{p} N} (1 + \phi)
    
    if p_hat == 1 or p_hat == 0:
        return 10e6
    
    return np.sqrt((1 - p_hat) * 1.8 / p_hat / N)

def adaptive_multilevel_subset_simulation(L, gamma, y_L):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    
    # output:
    # p_f: the probability of failure
    
    # y = y_l(gamma, y_L, L)
    # print("y: ", y)
    y = [-1.3, -2, -2.8, -3.3, y_L]
    tol = 0.03
    
    i = 0
    N_l = 0
    err = 10e6
    G_l = np.array([])
    while err > tol:
        i += 1
        N_l += 1
        G = np.random.normal(0, 1)
        kappa = k(G)
        # kappa = np.random.choice([-1, 1])
        G_l = np.append(G_l, G + kappa * gamma)
        
        mask = G_l <= y[0]
        p_hat = mask.mean()
        
        err = rRMSE(p_hat, N_l)
        
    p_f = p_hat
    # print("level: ", 1, "p_hat:", p_hat)
        
    for l in range(1, L):
        G_l = G_l[mask][:1]
        i = 0
        N_l = 1
        err = 10e6
        
        while err > tol:
            G_new = 0.8 * G_l[-1] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
            kappa_new = k(G_new)
            # kappa_new = np.random.choice([-1, 1])
            G_l_new = G_new + kappa_new * gamma ** (l+1)
            
            if G_l_new <= y[l-1]:
                G_l = np.append(G_l, G_l_new)
                i += 1
                N_l += 1
            else:
                G_l = np.append(G_l, G_l[-1])
                i += 1
                N_l += 1
                
            mask = G_l <= y[l]
            p_hat = mask.mean()
            
            # if i % 10 == 0:
            #     print("level: ", l+1, "p_hat: ", p_hat)
            
            err = rRMSE(p_hat, N_l)
            
        p_f *= p_hat
        # print("level: ", l+1, "p_hat:", p_hat)
        
    return p_f

if __name__ == "__main__":
    L = 5
    gamma = 0.5
    y_L = -3.8
    
    np.random.seed(0)
    start = time.time()
    p_f = adaptive_multilevel_subset_simulation(L, gamma, y_L)
    print("The failure probability is {:.2e}".format(p_f))
    print("Time: ", time.time() - start)
    
    failure_probabilities = [adaptive_multilevel_subset_simulation(L, gamma, y_L) for _ in range(1000)]
    print("The mean of the failure probability: ", np.mean(failure_probabilities))
    # print("The failure probabilities: ", failure_probabilities)
    
    from confidence_interval import bootstrap_confidence_interval
    
    # Calculate 95% confidence interval using bootstrap method
    confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.step(p_f, cdf, where='post')
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='90% Confidence Interval')
    plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()

# Solve the toy experiment using the adaptive multilevel subset simulation method.

import numpy as np
import time
import matplotlib.pyplot as plt

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

def adaptive_multilevel_subset_simulation(L, gamma, y_L, tol):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    
    # output:
    # p_f: the probability of failure
    
    y = [-1.3, -2, -2.8, -3.3, y_L]
    cost = np.zeros(L)
    
    i, err = 0, 10e6
    G_l = np.array([])
    N_l = 0
    while err > tol:
        i += 1
        N_l += 1
        cost[0] += 2
        G = np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        G_l = np.append(G_l, G + kappa * gamma)
        
        mask = G_l <= y[0]
        p_hat = mask.mean()
        
        err = rRMSE(p_hat, N_l)
        
    p_f = p_hat
    # print("level: ", 1, "p_hat:", p_hat)
        
    # level 2 to L
    for l in range(1, L):
        G_l = G_l[mask][:1]
        i = 0
        N_l = 1
        err = 10e6
        
        while err > tol:
            G_new = 0.8 * G_l[-1] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
            kappa_new = np.random.uniform(-1, 1)
            G_l_new = G_new + kappa_new * gamma ** (l+1)
            
            if G_l_new <= y[l-1]:
                G_l = np.append(G_l, G_l_new)
                i += 1
                N_l += 1
                cost[l] += 9 + l
            else:
                G_l = np.append(G_l, G_l[-1])
                i += 1
                N_l += 1
                cost[l] +=  9 + l
                
            mask = G_l <= y[l]
            p_hat = mask.mean()
            
            # if i % 10 == 0:
            #     print("level: ", l+1, "p_hat: ", p_hat)
            
            err = rRMSE(p_hat, N_l)
            
        p_f *= p_hat
        # print("level: ", l+1, "p_hat:", p_hat)
        
    return p_f, cost

if __name__ == "__main__":
    import csv
    L = 5
    gamma = 0.5
    y_L = -3.8
    failure_probabilities = []
    err_list = []
    cost_list = []

    for tol in [1, 0.5, 0.1, 0.05, 0.01]:
        np.random.seed(0)
        total_cost = []
        # with open("adaptive_multilevel_subsest_simulation_cost.csv", "a") as file:
        for _ in range(100):
            p_f, cost = adaptive_multilevel_subset_simulation(L, gamma, y_L, tol)
            # file.write(",".join(map(str, cost)) + "\n")
            failure_probabilities.append(p_f)
            total_cost.append(sum(cost))
    
        # with open('adaptive_multilevel_subsest_simulation_failure_probability.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for fp in failure_probabilities:
        #         writer.writerow([fp])
        
        ave = np.mean(failure_probabilities)
        print("The average probability of failure is: {:.2e}".format(ave))
        
        err = np.abs(ave - 7.23e-05) / 7.23e-05
        print("The relative error is: {:.2e}".format(err))
        err_list.append(err)
        
        cost = np.mean(total_cost)
        print("The average cost is: ", cost)
        cost_list.append(cost)
            
    # Define the principle line
    x = np.linspace(0.01, 0.5, 100)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(err_list, cost_list, marker='o')
    plt.loglog(x, 500 * x ** (-2), 'r--',  label=r'O($\epsilon^{-2}$)')
    plt.xlabel('Relative Error')
    plt.ylabel('Cost')
    plt.title('Adapter Multilevel Subset Simulation')
    plt.show()
    
    # start = time.time()
    # p_f, cost = adaptive_multilevel_subset_simulation(L, gamma, y_L)
    # print("Time: ", time.time() - start)
    # print("The failure probability is {:.2e}".format(p_f))
    # print("The cost is: ", cost) 
    
    # results = [adaptive_multilevel_subset_simulation(L, gamma, y_L) for _ in range(1000)]
    # failure_probabilities, costs = zip(*results)
    # print("The mean of the failure probability: ", np.mean(failure_probabilities))
    # # print("The failure probabilities: ", failure_probabilities)
    
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

    # relative_errors = [rRMSE(p, c) for p, c in results]
    
    # # Plot the complexity results
    # plt.figure(figsize=(8, 6))
    # plt.scatter(relative_errors, costs, alpha=0.5, marker='.')
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel('Relative Error')
    # plt.ylabel('Cost (Number of Samples)')
    # plt.title('Complexity Results: Relative Error vs Cost')
    # plt.grid(True)
    # plt.show()
    
# tol = 0.001  
# The average probability of failure is: 6.06e-05
# The relative error is: 6.21e-01
# The average cost is:  [29834.43 26084.43 16334.43  4671.36     0.  ]
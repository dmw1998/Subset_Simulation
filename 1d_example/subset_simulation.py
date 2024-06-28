# Subset simulation with fully refinement

from fenics import *
from kl_expansion import *
from mh_sampling import *
from IoQ import IoQ
from failure_probability import compute_cl
import numpy as np
import matplotlib.pyplot as plt

def subset_simulation(N, M, p0, u_max, n_grid, gamma = 0.8, L = 5):
    # input:
    # N: number of required samples
    # M: number of terms in the KL expansion
    # p0: failure probability
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    # L: number of levels
    
    # output:
    # p_f: failure probability
    
    # Initialize the list of the approximated IoQ
    G = []
    
    # Initialize the list of the initial states
    theta_ls = []
    
    p_f = p0
    
    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        theta_ls.append(thetas)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_max - u_1
        G.append(g)
    
    # Compute the threshold value
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 0, L)
    
    # print("Level: 0", "Threshold value: ", c_l)
    
    if c_l < 0:
        return len(G) / N
    
    # For l = 2, ..., L
    for l in range(1, L+1):
        # Generate N - N0 samples for each level
        G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma)
        
        # Compute the threshold value
        G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, l, L)
        
        # print("Level:", l, "Threshold value: ", c_l)
        
        if c_l < 0:
            break
        
        p_f *= p0
    
    return p_f * len(G) / N

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    n = len(data)
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, n), replace=True)
    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
    
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return lower_bound, upper_bound

if __name__ == "__main__":
    # Define the number of simulations
    num_simulations = 500
    
    # Define the number of samples
    N = 1000
    
    # Define the number of terms in the KL expansion
    M = 150
    
    # Define the failure probability
    p0 = 0.1
    
    # Define the upper bound of the solution
    u_max = 0.535
    
    # Define the number of mesh points
    n_grid = 512
    
    # Define the correlation parameter
    gamma = 0.8
    
    # Define the number of levels
    L = 5
    
    np.random.seed(1)
    # Compute the probability of failure
    p_f = subset_simulation(N, M, p0, u_max, n_grid, gamma, L)
    print("The probability of failure is: {:.2e}".format(p_f))
    
    # np.random.seed(0)
    # runs = 10
    # p_f = np.zeros(runs)
    # for i in range(0, runs):
    #     # print("Seed: ", i)
    #     # np.random.seed(i)
    #     # Compute the probability of failure
    #     p_f[i] = subset_simulation(N, M, p0, u_max, n_grid, gamma, L)
    #     print("The probability of failure is: {:.2e}".format(p_f[i]))
        
    #     # print("")
    
    # # save p_f
    # # np.save("p_f.npy", p_f)
    # print("The mean of the probability of failure is: {:.2e}".format(np.mean(p_f)))
    
    failure_probabilities = [subset_simulation(N, M, p0, u_max, n_grid, gamma, L) for _ in range(num_simulations)]
    # print("Failure probabilities:", failure_probabilities[0:10])

    # Calculate 95% confidence interval using bootstrap method
    confidence_interval = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=100, confidence_level=0.95)

    print("95% confidence interval for failure probability:", confidence_interval)
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Step 3: Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.xlim(1e-5, 1e-3)
    plt.step(p_f, cdf, where='post')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.grid(True)
    plt.show()
# A multileve estimator with fully refinement for the 1D example

from fenics import *
from kl_expansion import *
from IoQ import IoQ
from mh_sampling import *
from failure_probability import compute_cl
import numpy as np

def mle(p0, N, M, L_b, u_max = 0.535, n_grid = 4, L = 5):
    # input:
    # p0: probability of failure
    # N: number of samples
    # M: length of theta
    # L_b: burn-in length
    # u_max: critical value
    # n_grid: number of grid points
    # L: number of levels, L should larger than 3
    
    # output:
    # p_f: final probability of failure
    
    # Generate initial samples
    theta_ls = [np.random.randn(M) for _ in range(N)]
    G = [u_max - IoQ(kl_expan(theta), n_grid) for theta in theta_ls]
    
    # Determine the threshold value c_l
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 0, L)
    print('c_1 = ', c_l)
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return len(G) / N
    else:
        denominator = 1
        n_grid *= 2
    
    # l < 3, set L_b = 0    
    # Sampling without burn-in
    c_l_1 = c_l    # c_{l-1}
    G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 1, L)
    print('c_2 = ', c_l)
    
    G_, _ = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    
    denominator *= len([g for g in G_ if g <= c_l_1]) / N
    # print('denominator =', denominator)
    
    if c_l < 0:
        return p0 * len(G) / N / denominator
    
    n_grid *= 2
    
    # l > 2, set L_b = L_b
    # Sampling with burn-in
    for l in range(2, L):
        c_l_1 = c_l    # c_{l-1}
        G, theta_ls = sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
        G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, l, L)
        print('c_', l+1, ' = ', c_l)
        
        if c_l < 0:
            return p0 ** l * len(G) / N / denominator
        
        G_, _ = sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
        
        denominator *= len([g for g in G_ if g <= c_l_1]) / N
        # print('denominator =', denominator)
        
        n_grid *= 2
        
    G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    
    return p0 ** L * len(G) / N / denominator

if __name__ == "__main__":
    p0 = 0.25
    N = 1000
    M = 150
    L_b = 10
    u_max = 0.535
    n_grid = 4
    L = 8
    
    np.random.seed(0)
    p_f = mle(p0, N, M, L_b, u_max, n_grid, L)
    print("The probability of failure is: {:.2e}".format(p_f))
    
    from subset_simulation import bootstrap_confidence_interval
    
    failure_probabilities = [mle(p0, N, M, L_b, u_max, n_grid, L) for _ in range(100)]
    print("Failure probabilities:", failure_probabilities)

    # Calculate 95% confidence interval using bootstrap method
    confidence_interval = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=100, confidence_level=0.95)

    print("95% confidence interval for failure probability:", confidence_interval)
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Step 3: Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    # plt.xlim(1e-5, 1e-3)
    plt.step(p_f, cdf, where='post')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.grid(True)
    plt.show()
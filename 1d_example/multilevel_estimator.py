# A multileve estimator with fully refinement for the 1D example

from fenics import *
from kl_expansion import *
from IoQ import IoQ
from mh_sampling import *
from failure_probability import compute_cl
import numpy as np

def rRMSE(failure_probability):
    # input:
    # failure_probability
    
    # output:
    # relative rooted mean squared error
    
    failure_probability = np.array(failure_probability)
    
    difference = failure_probability - 1.6e-04
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 1.6e-04

def mle(N, p0 = 0.25, M = 150, L_b = 10, u_max = 0.535, n_grid = 32, L = 7):
    # input:
    # N: number of samples
    # p0: probability of failure
    # M: length of theta
    # L_b: burn-in length
    # u_max: critical value
    # n_grid: number of grid points
    # L: number of levels, L should larger than 3
    
    # output:
    # p_f: final probability of failure
    
    # Generate initial samples
    sample_numbers = np.zeros(L)
    theta_ls = [np.random.randn(M) for _ in range(N)]
    G = [u_max - IoQ(kl_expan(theta), n_grid) for theta in theta_ls]
    sample_numbers[0] += N
    
    # Determine the threshold value c_l
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 0, L)
    print('c_1 = ', c_l)
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return len(G) / N, sample_numbers
    else:
        denominator = 1
        n_grid *= 2
    
    # l < 3, set L_b = 0    
    # Sampling without burn-in
    c_l_1 = c_l    # c_{l-1}
    sample_numbers[1] += N * (1 - p0)
    G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 1, L)
    print('c_2 = ', c_l)
    
    sample_numbers[1] += N * (1 - p0)
    G_, _ = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    
    denominator *= len([g for g in G_ if g <= c_l_1]) / N
    # print('denominator =', denominator)
    
    # if c_l < 0:
    #     return p0 * len(G) / N / denominator, sample_numbers
    
    n_grid *= 2
    
    # l > 2, set L_b = L_b
    # Sampling with burn-in
    for l in range(2, L):
        c_l_1 = c_l    # c_{l-1}
        sample_numbers[l] += N + (L_b - 1) * N * p0
        G, theta_ls = sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
        G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, l, L)
        print('c', l+1, ' = ', c_l)
        
        # if c_l < 0:0 sample_numbers
        
        sample_numbers[l] += N + (L_b - 1) * N * p0
        G_, _ = sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
        
        denominator *= len([g for g in G_ if g <= c_l_1]) / N
        # print('denominator =', denominator)
        
        n_grid *= 2
    
    sample_numbers[L-1] += N + (L_b - 1) * N * p0
    G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    
    return p0 ** L * len(G) / N / denominator, sample_numbers

if __name__ == "__main__":
    p0 = 0.25
    N = 1000
    M = 150
    L_b = 10
    u_max = 0.535
    n_grid = 16
    L = 7

    failure_probability = []
    np.random.seed(32)
    for i in range(10):
        p_f, samples = mle(N, p0 = 0.25, M = 150, L_b = 10, u_max = 0.535, n_grid = n_grid, L = L)
        failure_probability.append(p_f)
        print("{:.0f}: {:.2e}".format(i, p_f))
        print("samples: ", samples)
        print("error: {:.2e}".format(rRMSE(p_f)))
        
    ave = np.mean(failure_probability)
    print("The probability of failure is: {:.2e}".format(ave))
    err = rRMSE(failure_probability)
    print("The error is {:.2e}".format(err))
        
    # from subset_simulation import bootstrap_confidence_interval
    
    # np.random.seed(0)
    # failure_probabilities = [mle(N, p0, M, L_b, u_max, n_grid, L) for _ in range(100)]
    # print("Failure probabilities:", failure_probabilities)

    # # Calculate 95% confidence interval using bootstrap method
    # confidence_interval = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=100, confidence_level=0.95)

    # print("95% confidence interval for failure probability:", confidence_interval)
    
    # p_f = sorted(failure_probabilities)
    # cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # # Step 3: Plot the empirical CDF
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
    # # plt.xlim(1e-5, 1e-3)
    # plt.step(p_f, cdf, where='post')
    # plt.xlabel('Probability')
    # plt.ylabel('Empirical CDF')
    # plt.title('Empirical CDF of Probabilities')
    # plt.grid(True)
    # plt.show()
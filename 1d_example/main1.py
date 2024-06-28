# Subset simulation with fully refinement

from fenics import *
from kl_expansion import *
from mh_sampling import *
from IoQ import IoQ
from failure_probability import compute_cl
import numpy as np
import matplotlib.pyplot as plt

def main(L_b, N, M, p0, u_max, n_grid, gamma = 0.8, L = 5):
    # input:
    # N: number of required samples
    # M: number of terms in the KL expansion
    # p0: failure probability
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    # L: number of levels
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    # Initialize the list of the approximated IoQ
    G0 = []
    G1 = []
    
    # Initialize the list of the initial states
    theta_ls0 = []
    
    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        theta_ls0.append(thetas)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_max - u_1
        G0.append(g)
        # u_1 = IoQ(kl_expansion_corr(thetas), n_grid)
        # g = u_max - u_1
        # G1.append(g)
    
    # For subset simulation with fully refinement
    G = G0
    theta_ls = theta_ls0
    p_f_sub = p0
    
    # Compute the threshold value
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 1, L)
    
    print("Level: 0", "Threshold value: ", c_l)
    
    if c_l < 0:
        p_f_sub = len(G) / N
    else:
        # For l = 2, ..., L
        for l in range(1, L+1):
            # Generate N - N0 samples for each level
            G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma)
            
            # Compute the threshold value
            G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, l, L)
            
            print("Level:", l, "Threshold value: ", c_l)
            
            if c_l < 0:
                break
            
            p_f_sub *= p0
        
        p_f_sub *= len(G) / N
    
    # For multilevel estimator
    G = G0
    theta_ls = theta_ls0
    p0 = 0.25
    L = 7
    n_grid = 4
    
    # Determine the threshold value c_l
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 0, L)
    print('c_1 = ', c_l)
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return p_f_sub, len(G) / N
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
            return p_f_sub, p0 ** l * len(G) / N / denominator
        
        G_, _ = sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
        
        denominator *= len([g for g in G_ if g <= c_l_1]) / N
        # print('denominator =', denominator)
        
        n_grid *= 2
        
    G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8)
    
    return p_f_sub, p0 ** L * len(G) / N / denominator

if __name__ == "__main__":
    p0 = 0.1
    N = 100
    M = 10
    L_b = 10
    u_max = 0.535
    n_grid = 100
    
    np.random.seed(0)
    p_f_sub, p_f_mle = main(L_b, N, M, p0, u_max, n_grid)
    print("The probability of failure by subset simulation is: {:.2e}".format(p_f_sub))
    print("The probability of failure by multilevel estimator is: {:.2e}".format(p_f_mle)
    )
    
    print("")
    
    np.random.seed(1)
    p_f_sub, p_f_mle = main(L_b, N, M, p0, u_max, n_grid)
    print("The probability of failure by subset simulation is: {:.2e}".format(p_f_sub))
    print("The probability of failure by multilevel estimator is: {:.2e}".format(p_f_mle)
    )
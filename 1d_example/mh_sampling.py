# Using the Metropolis-Hastings algorithm to sample from the posterior distribution

from fenics import *
from kl_expansion import kl_expan
from IoQ import IoQ
from failure_probability import compute_cl
import numpy as np
import matplotlib.pyplot as plt

def sampling_one_new_theta(G, theta_1, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # G: the approximated IoQ
    # theta_1: current state
    # c_l: threshold value
    # u_max: critial value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_new: new state
    
    M = len(theta_1)
    # theta_c = np.zeros_like(theta_1)
    # for i in range(M):
    #     theta = theta_1[i]
        
    #     theta_tilde = gamma * theta + np.sqrt(1 - gamma**2) * np.random.normal()
        
    #     # Compute the ratio r = f(theta_tilde) / f(theta), 
    #     # where f is the pdf of the Gaussian distribution
    #     # f(x) = exp(-0.5 * x^2) / sqrt(2 * pi)
    #     # r = exp(-0.5 * (theta_tilde**2 - theta**2))
    #     r = exp(0.5 * (theta**2 - theta_tilde**2))
        
    #     if np.random.rand() < min(1, r):
    #         theta_c[i] = theta_tilde
    #     else:
    #         theta_c[i] = theta
    
    theta_c = gamma * theta_1 + np.sqrt(1 - gamma**2) * np.random.randn(M)
        
    # Solve the PDE for required mesh size
    u_h = IoQ(kl_expan(theta_c), n_grid)
    
    # Acceptance condition
    if u_max - u_h <= c_l:
        # If theta_c in F_{l-1}, then accept theta_c
        theta_new = theta_c
        G_new = (u_max - u_h)
    else:
        # If theta_c not in F_{l-1}, then reject theta_c
        theta_new = theta_1
        G_new = G[-1]
    
    return G_new, theta_new

def sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # c_l: threshold value
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N - N0:
        theta = theta_ls[i]
        G_new, theta_new = sampling_one_new_theta(G, theta, c_l, u_max, n_grid, gamma = gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
    
    return G, theta_ls

def sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # L_b: burn-in length
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # c_l: threshold value
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N + (L_b - 1) * N0:
        theta = theta_ls[i]
        G_new, theta_new = sampling_one_new_theta(G, theta, c_l, u_max, n_grid, gamma = gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
        
    G = G[N0 * L_b:]
    theta_ls = theta_ls[N0 * L_b:]
    
    return G, theta_ls

if __name__ == "__main__":
    # Define the number of terms in the KL expansion
    M = 150
    
    # Define the number of samples
    N = 10
    
    # Define the critical value
    u_max = 0.535
    
    # Define the number of mesh points
    n_grid = 100
    
    # Define the correlation parameter
    gamma = 0.8
    
    # Define the initial state
    thetas = np.random.normal(0, 1, M)
    
    # Compute the random field
    a_x = kl_expan(thetas)
    
    # Compute the IoQ
    u_h = IoQ(a_x, n_grid)
    
    # Initialize the list of the approximated IoQ
    G = [u_max - u_h]
    
    # Initialize the list of the thetas
    thetas_list = [thetas]
    
    for i in range(N):
        G_new, thetas = sampling_one_new_theta(G, thetas_list[i], 0.3, u_max, n_grid, gamma)
        G.append(G_new)
        thetas_list.append(thetas)
        
    G, thetas_list, c_l = compute_cl(G, thetas_list, N, 0.3, 0, 6)
    
    print('c_0=', c_l)
    
    # Sample from the posterior distribution
    G, thetas_list = sampling_theta_list(N, G, thetas_list, c_l, u_max, n_grid, gamma = gamma)
    
    print('G =', len(G))
    print('thetas =', len(thetas_list))
    
    G, thetas_list, c_l = compute_cl(G, thetas_list, N, 0.1, 1, 6)
    
    print('c_1=', c_l)
    
    G, thetas_list = sampling_theta_burn_in(10, N, G, thetas_list, c_l, u_max, n_grid, gamma = gamma)
    
    print('G =', len(G))
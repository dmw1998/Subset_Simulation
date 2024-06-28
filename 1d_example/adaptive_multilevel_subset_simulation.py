# multilevel subset simulation with selective refinement

from fenics import *
from compute_yl import y_l
from compute_rRMSE import rRMSE
from kl_expansion import *
from mh_sampling import *
from IoQ import IoQ
from compute_correlation_factor import compute_autocorrelation_multidimensional
import numpy as np
import matplotlib.pyplot as plt
import time


def adaptive_multilevel_subset_simulation(M, u_max, n_grid, gamma, corr_coeff = 0.8, L = 5):
    # input:
    # M: number of terms in the KL expansion
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: accuracy parameter
    # corr_coeff: correlation parameter
    # L: number of levels
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    # The sequence of the threshold values
    y = y_l(L, gamma)
        
    # print("y: ", y)
    # print(" ")
    
    i = 0
    N = 0
    n = 0
    theta_ls = []
    G = []
    i_ls = []
    delta = 10**6   # delta = infinty
    while delta > 10**(-4):
        theta = np.random.normal(0, 1, M)
        theta_ls.append(theta)
        u_1 = IoQ(kl_expan(theta), n_grid)
        g = u_max - u_1
        G.append(g)
        print("g: ", g)
        i += 1
        N += 1
        # time.sleep(1)
        
        if g < y[0]:
            # if g is in the failure domain F_{1}
            # else delta = infinty
            i_ls.append(i)
            n += 1
            
            # for the independent random variables
            # correlation factor is 0
            delta = rRMSE(n/N, 0, N)
            print("delta: ", delta)
            # time.sleep(1)
            
    p_f = n / N
    print("p_f: ", p_f)
    
            
    for l in range(1, L):
        i = 0
        N = 0
        n = 0
        print(" ")
        print("y_", l, ": ", y[l])
        
        ind = i_ls[0]
        theta0 = theta_ls[ind]        # initial state theta_{l,0} = theta_{l-1,i0}
                                    # where i0 is the first index of the failure sample
        
        if G[ind] < y[l]:
            # if g is in the failure domain F_{l+1}
            p_l = 1
            # autocorr = compute_autocorrelation_multidimensional(theta_ls)
            delta = rRMSE(p_l, 0, N)
            # delta = rRMSE(p_l, 0, 1)
            print("delta: ", delta)
            
            # initialize the list of the samples and the approximated IoQ
            # theta0 is in F_{l+1}, the first sample in the failure domain
            theta_ls = [theta0]
            G = [G[0]]
            N = 1
            n = 0
            
        else:
            # if g is not in falure domain F_{l+1}
            # delta = infinty
            p_l = 0
            delta = 10**6
            
            # initialize the list of the samples and the approximated IoQ
            theta = np.random.normal(0, 1, M)
            u_1 = IoQ(kl_expan(theta), n_grid)
            g = u_max - u_1
            theta_ls = [theta]
            G = [g]
            N = 1
            n = 0
        
        i_ls = []
        while delta > 10**(-4):
            # print("iteration: ", i)
            if i > 1000:
                break
            
            # sampling a new theta
            theta_c = gamma * theta0 + np.sqrt(1 - gamma**2) * np.random.randn(M)
            g = u_max - IoQ(kl_expan(theta_c), n_grid)
            # print("g: ", g)
            # time.sleep(1)
            
            i += 1
            N += 1
            
            if g < y[l]:
                # if g is in the failure domain F_{l}
                n += 1
                # print(n)
                theta0 = theta_c
                theta_ls.append(theta0)
                G.append(g)
                
            if n == 0:
                continue
            else:
                autocorr = compute_autocorrelation_multidimensional(theta_ls)
                delta = rRMSE(n/N, autocorr, N)
                # delta = rRMSE(n/N, 0, N)
                # print("delta: ", delta)
        
        # print(g)
        print("i: ", i) 
        print("n: ", n)
        print("N: ", N)   
        if n == 0:
            continue
        else: 
            p_f *= n/N
        print("p_f: ", p_f)
        print(" ")
        
    return p_f
                
if __name__ == "__main__":
    M = 150
    p0 = 0.1
    u_max = 0.535
    n_grid = 100
    gamma = 0.15
    L = 5
    
    np.random.seed(0)
    p_f = adaptive_multilevel_subset_simulation(M, u_max, n_grid, gamma, L = L)
    print(" ")
    print("final probability{:.2e}".format(p_f))

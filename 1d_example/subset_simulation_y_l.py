# An idea

from fenics import *
from compute_yl import y_l
from kl_expansion import *
from mh_sampling import *
from IoQ import IoQ
import numpy as np

def subset_simulation_yl(N, M, u_max, n_grid, gamma, corr_coeff = 0.8, L = 5):
    # input:
    # N: number of required samples
    # M: number of terms in the KL expansion
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: accuracy parameter
    # corr_coeff: correlation parameter
    # L: number of levels
    
    # output:
    # p_f: failure probability
    
    s = True
    
    # Initialize the threshold values
    # y =[0.021068347431547174, 0.013600918999687628, 0.0046197026089659365, 10e-6]
    # L = len(y)
    y = y_l(L, gamma)
    
    # Initialize the list of the approximated IoQ
    G = []
    
    # Initialize the list of the initial states
    theta_ls = []
    
    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_max - u_1
        theta_ls.append(thetas)
        G.append(g)
            
    theta_new = []
    G_new = []
    for index, value in enumerate(G):
        if value < y[0]:
            theta_new.append(theta_ls[index])
            G_new.append(G[index])
            
    theta_ls = theta_new
    G = G_new
    
    p_f = len(G) / N
    # print("prob at 1 is", p_f)
    
    for l in range(1, L):
        # Generate N - N0 samples for each level
        G_c, theta_c_ls = sampling_theta_list(N, G, theta_ls, y[l-1], u_max, n_grid, corr_coeff)
        
        theta_ls = []
        G = []
        for i in range(N):
            g = G_c[i]
            if g < y[l]:
                theta_ls.append(theta_c_ls[i])
                G.append(g)
        
        k = 0     
        while len(G) == 0:
            # Resample the initial states
            theta_ls = [theta_c_ls[-1]]
            G = [G_c[-1]]
            G_c, theta_c_ls = sampling_theta_list(N, G, theta_ls, y[l-1], u_max, n_grid, corr_coeff)
            
            theta_ls = []
            G = []
            for i in range(N):
                g = G_c[i]
                if g < y[l]:
                    theta_ls.append(theta_c_ls[i])
                    G.append(g)
            k += 1
            # print("Zero probability of failure at level:", l, "i:", k)
            
            if k > 1000:
                s = False
                print("Zero probability of failure at level:", l, "i:", k)
                return s, p_f
        
        # print("prob at", l+1, "is", len(G) / N)
        p_f *= len(G) / N
        
    return s, p_f

if __name__ == "__main__":
    N = 1000
    M = 150
    u_max = 0.535
    n_grid = 512
    gamma = 0.13
    corr_coeff = 0.8
    L = 5
    
    y = y_l(L, gamma)
    print("y: ", y)
    
    np.random.seed(0)
    s, p_f = subset_simulation_yl(N, M, u_max, n_grid, gamma, corr_coeff, L)
    print("failure probability  {:.2e}".format(p_f))
    s = False
    p_f = np.zeros(500)
    for i in range(500):
        ind = i
        while True:
            s, p = subset_simulation_yl(N, M, u_max, n_grid, gamma, corr_coeff, L)
            # print("s", s)
            if s:
                p_f[ind] = p
                print("failure probability {:.2e}".format(p_f[ind]))
                break
        
    # p_f_mean = np.mean(p_f)
    # print("mean of failure probability {:.2e}".format(p_f_mean))
    
    from subset_simulation import bootstrap_confidence_interval

    # Calculate 95% confidence interval using bootstrap method
    confidence_interval = bootstrap_confidence_interval(p_f, num_bootstrap_samples=100, confidence_level=0.95)

    print("95% confidence interval for failure probability:", confidence_interval)
    
    p_f = sorted(p_f)
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
from kl_expansion import kl_expan
from IoQ import IoQ
from compute_yl import y_l
from compute_rRMSE import rRMSE

import numpy as np

N = 100
M = 150
L = 5
gamma = 0.15
u_max = 0.535
n_grid = 100
p_f = 1

y = y_l(L, gamma)

i = 0
N_l = 1
delta = 10**6

while delta > 1:
    theta_ls = []
    G = []
    
    for i in range(N):
        theta = np.random.normal(0, 1, M)
        g = u_max - IoQ(kl_expan(theta), n_grid)
        theta_ls.append(theta)
        G.append(g)
        
    mask = np.array(G) < y[0]
    p_l = np.sum(mask) / N
    
    delta = rRMSE(p_l, 0, N_l)


# the first index that G < y[0]
i0 = np.where(mask == True)[0][0]
p_f *= p_l

for l in range(1, L):
    print("y[", l, "]: ", y[l])
    i = 1
    N_l = 1
    
    mask = np.array(G) < y[l]
    p_l = np.sum(mask) / N
    delta = rRMSE(p_l, 0.8, N_l)
    print(delta)
    
    # theta_ls = [theta_ls[i0]]
    # G = [G[i0]]
    
    while delta > gamma**l:
        if i > 1000:
            break
        
        theta_ls = [theta_ls[i0]]
        G = [G[i0]]
        for i in range(N - i0):
            theta_c = 0.8 * theta_ls[i] + np.sqrt(1 - 0.8**2) * np.random.normal(0, 1, M)
            g_c = u_max - IoQ(kl_expan(theta_c), n_grid)
            theta_ls.append(theta)
            G.append(g)
        
        i += 1
        N_l += 1
        
        mask = np.array(G) < y[l]
        
        p_l = np.sum(mask) / N
        print(p_l)
        delta = rRMSE(p_l, 0.8, N_l)
        print(delta)
    
    i0 = np.where(mask == True)[0][0]
    print("i0", i0)
    p_f *= p_l
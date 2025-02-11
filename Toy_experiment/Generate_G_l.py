
# Generate G_l(\omega) = G(\omega) + \kapa(\omega) * \gamma^{l} with \kapa(\omega) ~ U({-1, 1}) and \gamma = 0.5

import numpy as np

def Generate_G_l(G_l, theta, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    # l: current level
    # c_l: probability threshold for each subset
    # gamma: auto-correlation factor
    
    # output:
    # G_l: new samples for next level
    # kappa: new samples for next level
    
    N0 = len(G_l)
    
    for i in range(N - N0):
        theta_new = 0.8 * G_l[i]  + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa_new = np.random.choice([-1, 1])
        G_l_new = theta_new + kappa_new * (gamma ** l)
        
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
            theta = np.append(theta, theta_new)
        else:
            G_l = np.append(G_l, G_l[i])
            theta = np.append(theta, theta[i])
            
    return G_l, theta

if __name__ == "__main__":
    # Parameters
    N = 1000  # Total number of samples per level
    p_0 = 0.1  # Probability threshold for each subset
    gamma = 0.5
    l = 1  # Level
    y_L = -3.8  # Failure threshold
    
    # Generate initial samples for level 0
    G = np.random.normal(0, 1, N)       # theta
    kappa = np.random.choice([-1, 1], N)
    G_l = G + kappa * (gamma ** l)
    
    # find c_l
    c_l = np.percentile(G_l, 100 * p_0)
    print("c_l: ", c_l)
    
    mask = G_l <= c_l
    print("Number of samples in failure domain: ", np.sum(mask))
    
    G_l = G_l[mask]
    G = G[mask]
    
    l += 1
    
    G_l = Generate_G_l(G_l, G, N, l, c_l, gamma = gamma)
    
    c_l = np.percentile(G_l, 100 * p_0)
    print("c_l: ", c_l)
# Multilevel estimator for the toy experiment

import numpy as np

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    
    difference = p_hat - 7.23e-05
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 7.23e-05

def sample_new_G(G, G_l, N, l, c_l, gamma = 0.5):
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
        # Propose a new sample for G ~ N(0,1)
        G_new = 0.8 * G[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        
        # Noise
        # kappa_new = np.random.choice([-1, 1])
        kappa_new = np.random.uniform(-1, 1)
        
        # Compute the new G_l
        G_l_new = G_new + kappa_new * gamma ** l      
          
        # Check if the new sample is in the failure domain
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
            G = np.append(G, G_new)
        else:
            G_l = np.append(G_l, G_l[i])
            G = np.append(G, G[i])
            
    return G_l, G


def mle(N, L_b = 3, y_L = -3.8, p0 = 0.25, gamma = 0.5, L = 7):
    # input:
    # N: total number of samples per level
    # L_b: burn-in length
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    N0 = int(N * p0)
    
    # Initialization, l= 1
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma
    cost = N * gamma ** (-2)
    
    # Compute the probability threshold
    c_l = sorted(G_l)[N0-1]
    # print("The probability threshold for level 1 is", c_l)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        # print("left at level 1")
        return np.mean(mask) / N, cost
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    G = G[mask][:N0]
    
    dinominator = 1
    
    # level 2, no burn-in
    G_l, G = sample_new_G(G, G_l, N, 2, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-4)
    c_l_1 = c_l
    
    c_l = sorted(G_l)[N0-1]
    # print("The probability threshold for level 2 is", c_l)
    
    if c_l <= y_L:
        # print("left at level 2")
        mask = G_l <= y_L
        return p0 * np.mean(mask) / dinominator, cost
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    G = G[mask][:N0]
    
    G_l_1, _ = sample_new_G(G, G_l, N, 2, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-4)
    mask = G_l_1 <= c_l_1
    dinominator *= np.mean(mask)
    # print("The denominator for level 2 is", dinominator)
    
    # level > 2, with burn-in
    N += L_b * N0
    for l in range(3, L):
        G_l, G = sample_new_G(G, G_l, N, l, c_l, gamma = gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        # drop the first L_b samples in each Markov chain
        G_l = G_l[int(N0*L_b):]
        G = G[int(N0*L_b):]
        c_l_1 = c_l
        
        c_l = sorted(G_l)[N0]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            # print("left at level", l)
            return p0 ** (l-1) * np.sum(mask) / N / dinominator, cost

        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        G = G[mask][:N0]
        
        G_l_1, _ = sample_new_G(G, G_l, N, l, c_l, gamma = gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        # drop the first L_b samples in each Markov chain
        G_l_1 = G_l_1[int(N0*L_b):]
        mask = G_l_1 <= c_l_1
        dinominator *= np.mean(mask)
        # print("The denominator for level", l, "is", dinominator)
        
    G_l, _ = sample_new_G(G, G_l, N, L, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-2 * L)
    # drop the first L_b samples in each Markov chain
    G_l = G_l[int(N0*L_b):]
    mask = G_l <= y_L
    # print("reach the last level")
    
    return p0 ** (L-1) * np.mean(mask) / dinominator, cost

def run_simulation(args):
    N, seed = args
    np.random.seed(seed)
    p_f, cost = mle(N)
    return p_f, cost

if __name__ == "__main__":
    import multiprocessing as mp
    from tqdm import tqdm
    
    err_list = []
    cost_list = []
    
    np.random.seed(199)
    for N in [500, 1000, 1500, 3500, 7000, 100000]:
        failure_probabilities = []
        cost = []
        seeds = np.random.randint(0, 1000, 100)
        
        with mp.Pool(processes=12) as pool:
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, args), total=100, desc=f"N = {N}"))
            
            failure_probabilities = [res[0] for res in results]
            cost = [res[1] for res in results]
            
        p = np.mean(failure_probabilities)
        print("The failure probability is {:.2e}".format(p))
        cost_list.append(np.mean(cost))
        print("The cost is {:.2e}".format(np.mean(cost)))
        err = rRMSE(failure_probabilities)
        print("The relative error is {:.2e}\n".format(err))
        err_list.append(err)
        
        np.save("MLE_cost.npy", cost_list)
        np.save("MLE_error.npy", err_list)
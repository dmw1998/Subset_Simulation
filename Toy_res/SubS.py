import numpy as np

fix_level = 5

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    
    difference = p_hat - 7.23e-05
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 7.23e-05

def sample_new_G(G_l, G, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: G_l <= c_l
    # G: samples in the failure domain
    
    # output:
    # G_l: new G_l <= c_l
    # G: new samples for next level
    
    N0 = len(G_l)
    rho=0.8
    
    for i in range(N - N0):
        # Propose a new sample for G ~ N(0,1)
        G_new = rho * G_l[i] + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1)
        
        # Noise
        kappa_new = np.random.uniform(-1, 1)
        
        # Compute the new G_l
        G_l_new = G_new + kappa_new * gamma ** fix_level    
          
        # Check if the new sample is in the failure domain
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
            G = np.append(G, G_new)
        else:
            G_l = np.append(G_l, G_l[i])
            G = np.append(G, G[i])
            
    # print("the acceptance rate at level {} is {:.2f}".format(l, r / (N - N0)))
    
    return G_l, G

def subset_simulation(N, y_L = -3.8, p0 = 0.2, gamma = 0.5, L = 6):
    # input:
    # N: total number of samples per level
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: error factor
    # L: total number of levels
    
    N0 = int(N * p0)
    cost = np.zeros(L)
    
    # Initialization, l= 1
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma ** fix_level
    cost[0] = N * gamma ** (-2 * fix_level)
    
    for l in range(2, L):
        c_l = np.percentile(G_l, 100 * p0)
        
        if c_l < y_L:
            mask = G_l <= y_L
            p_fail = p0 ** (l-1) * np.sum(mask) / N
            return p_fail, cost

        mask = G_l <= c_l
        
        G_l = G_l[mask][:N0]
        G = G[mask][:N0]
        
        # Generate new samples for next level
        G_l, G = sample_new_G(G_l, G, N, l+1, c_l, gamma = gamma)
        cost[l] = (N - N0) * gamma ** (-2 * fix_level)
        
    # Final level
    mask = G_l <= y_L
    p_fail = p0 ** (L-1) * np.mean(mask)
        
    return p_fail, cost

def run_simulation(args):
    N, seed = args
    np.random.seed(seed)
    p_f, cost = subset_simulation(N)
    return p_f, np.sum(cost)

if __name__ == "__main__":
    import multiprocessing
    from tqdm import tqdm
    
    error_list = []
    cost_list = []
    
    np.random.seed(10)
    for N in [1000, 5000, 8000, 10000, 50000, 70000, 100000]:
    # for N in [100000, 150000, 200000]:
        failure_probability = []
        cost = []
        seeds = np.random.randint(100, 1000, 100)
        
        with multiprocessing.Pool(12) as pool:
            arges = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, arges), total=100, desc=f"N = {N}"))
            
        for res in results:
            failure_probability.append(res[0])
            cost.append(res[1])

        err = rRMSE(failure_probability)
        print("The relative root mean square error is {:.2e}".format(err))
        print("The average cost is {:.2e}\n".format(np.mean(cost)))
        error_list.append(err)
        cost_list.append(np.mean(cost))
        
        np.save("SubS_error_list.npy", error_list)
        np.save("SubS_cost_list.npy", cost_list)
        
    # error_list = np.load("SubS_error_list.npy").tolist()
    # cost_list = np.load("SubS_cost_list.npy").tolist()
    
    # np.random.seed(140)
    # for N in [100500, 101000, 105000, 108000, 110000, 150000, 170000, 200000]:
    #     failure_probability = []
    #     cost = []
    #     seeds = np.random.randint(100, 1000, 100)
        
    #     with multiprocessing.Pool(12) as pool:
    #         arges = [(N, seed) for seed in seeds]
    #         results = list(tqdm(pool.imap(run_simulation, arges), total=100, desc=f"N = {N}"))
            
    #     for res in results:
    #         failure_probability.append(res[0])
    #         cost.append(res[1])

    #     err = rRMSE(failure_probability)
    #     print("The relative root mean square error is {:.2e}".format(err))
    #     print("The average cost is {:.2e}\n".format(np.mean(cost)))
    #     error_list.append(err)
    #     cost_list.append(np.mean(cost))
        
    #     np.save("SubS_error_list.npy", error_list)
    #     np.save("SubS_cost_list.npy", cost_list)
    
    # N = 100000
    # failure_probability = []
    # for i in range(100):
    #     print(f"Simulation {i}")
    #     p_f, cost = subset_simulation(N)
    #     print("The failure probability is {:.2e}".format(p_f))
    #     failure_probability.append(p_f)
    #     # print("The cost is {:.2e}".format(np.sum(cost)))
    #     print("The error is {:.2e}\n".format(rRMSE([p_f])))
    # print("The average failure probability is {:.2e}".format(np.mean(failure_probability)))
    # print("The error is {:.2e}".format(rRMSE(failure_probability)))
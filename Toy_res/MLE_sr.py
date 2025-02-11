import numpy as np

def generate_samples(w, G, N, gamma, c, l, burn_in=0):
    N0 = len(w)
    L_b = burn_in * N0
    cost = 0
    for i in range(N + (burn_in - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        # cost += 7
        kappa = np.random.uniform(-1, 1)
        add_term = kappa * gamma
        tol = gamma
        G_new = w_new + add_term
        # cost += 2
        cost += gamma ** (-2)
        for j in range(2, l):
            if tol >= np.abs(G_new - c):
                tol *= gamma
                add_term *= gamma
                G_new = w_new + add_term
                cost += gamma ** (-2 * j)
            else:
                break
            
        if G_new <= c:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
        # cost += 1
            
    return w[L_b:], G[L_b:], cost

def mle_sr(N, gamma=0.5, y=-3.8, p_0=0.25, L=7, burn_in=3):
    
    N0 = int(p_0 * N)
    
    l = 1
    w = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G = w + kappa * gamma
    cost = N * gamma ** (-2)
    
    c_1 = np.sort(G)[N0-1]
    
    # For l = 2, no burn-in
    mask = G <= c_1
    w = w[mask][:N0]
    G = G[mask][:N0]
    
    w, G, add_cost = generate_samples(w, G, N, gamma, c_1, 2)
    
    cost += add_cost
    
    c_2 = np.sort(G)[N0-1]
    _, G_2, add_cost = generate_samples(w, G, N, gamma, c_2, 2)
    
    cost += add_cost
    
    mask = G_2 <= c_1
    denominator = np.mean(mask)
    
    # For l > 2, burn-in
    c_l = c_2
    for l in range(3,L):
        c_l_1 = c_l
        w, G, add_cost = generate_samples(w, G, N, gamma, c_l, l, burn_in)
        cost += add_cost
        c_l = np.sort(G)[N0-1]
        
        if c_l <= y:
            mask = G <= y
            failure_probability = p_0 ** (l-1) * np.mean(mask) / denominator
            return failure_probability, cost
        
        mask = G <= c_l
        w = w[mask][:N0]
        G = G[mask][:N0]
        
        _, G_l, add_cost = generate_samples(w, G, N, gamma, c_l, l, burn_in)
        cost += add_cost
        
        mask = G_l <= c_l_1
        denominator *= np.mean(mask)
        
    # For l = L
    c_L_1 = c_l
    w, G, add_cost = generate_samples(w, G, N, gamma, c_l, L, burn_in)
    cost += add_cost
    
    mask = G <= y
    w = w[mask]
    G = G[mask]
    p_L = np.mean(mask)
    
    _, G_L, add_cost = generate_samples(w, G, N, gamma, y, L, burn_in)
    cost += add_cost
    
    mask = G_L < c_L_1
    denominator *= np.mean(mask)
    
    failure_probability = p_0 ** (L-1) * p_L / denominator
    
    return failure_probability, cost

def run_simulation(args):
    N, seed = args
    np.random.seed(seed)
    p_f, cost = mle_sr(N)
    return p_f, np.sum(cost)

if __name__ == "__main__":
    import multiprocessing
    from tqdm import tqdm
    cost_list = []
    err_list = []
    np.random.seed(0)
    
    for N in [80, 100, 500, 1000, 3000, 5000]:
        failure_probabilities = []
        costs = []
        seeds = np.random.randint(0, 1000, 100)
        
        with multiprocessing.Pool(12) as pool:
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, args), total=100, desc=f"N = {N}"))
            
        for res in results:
            failure_probabilities.append(res[0])
            costs.append(res[1])
        
        ave = np.mean(failure_probabilities)
        print("The average probability of failure is: {:.2e}".format(ave))
        total_cost = np.mean(costs)
        print("The mean cost: {:.2e}".format(total_cost))
        cost_list.append(total_cost)
        
        err = 1/7.23e-05*np.sqrt(np.mean((np.array(failure_probabilities) - 7.23e-05)**2))
        print("The relative error: {:.2e}\n".format(err))
        err_list.append(err)
        
        np.save("MLE_sr_error_list", err_list)
        np.save("MLE_sr_cost", cost_list)

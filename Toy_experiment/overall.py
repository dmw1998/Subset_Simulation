import numpy as np
import matplotlib.pyplot as plt

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    
    difference = p_hat - 7.23e-05
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 7.23e-05
    
    # ave = np.abs(p_hat - 7.23e-05)
    
    # return np.mean(ave) / 7.23e-05

def sample_new_G_ss(G_l, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    
    # output:
    # G_l: new samples for next level
    
    N0 = len(G_l)
    
    for i in range(N - N0):
        # Propose a new sample for G ~ N(0,1)
        # Computational cost: 6
        G_new = 0.8 * G_l[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        # Add noise
        kappa_new = np.random.uniform(-1, 1)
        # Compute the new G_l
        # Computational cost: 2 + l
        G_l_new = G_new + kappa_new * gamma ** l        
        # Computational cost: 1
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
        else:
            G_l = np.append(G_l, G_l[i])
            
    # Computational cost in total: (9 + l) * N0
            
    return G_l

def classical_subset_simulation(N, y_L = -3.8, p0 = 0.1, gamma = 0.5, L = 5):
    # input:
    # N: total number of samples per level
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    N0 = int(N * p0)
    
    # Initialization, l= 1
    # Computational cost: 2N
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma
    cost = N * gamma ** (-2)
    
    # Compute the probability threshold
    c_l = sorted(G_l)[N0-1]
    # print("The probability threshold for level 1 is", c_l)
    
    # if c_l <= y_L:
    #     mask = G_l <= y_L
    #     return np.mean(mask), cost
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    for l in range(2, L):
        G_l = sample_new_G_ss(G_l, N, l, c_l, gamma = gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        
        c_l = sorted(G_l)[N0-1]
        # print("The probability threshold for level", l, "is", c_l)
        
        # if c_l <= y_L:
        #     mask = G_l <= y_L
        #     return p0 ** (l-1) * np.mean(mask), cost
        
        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        
    G_l = sample_new_G_ss(G_l, N, L, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-2 * L)
    mask = G_l <= y_L
    # print("The number of samples in the failure domain is", np.sum(mask))
    return p0 ** (L-1) * np.mean(mask), cost

def sample_new_G(G_l, N, l, c_l, gamma = 0.5):
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
        G_l_new = 0.8 * G_l[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa_new = np.random.uniform(-1, 1)
        G_l_new += kappa_new * (gamma ** l)
        
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
        else:
            G_l = np.append(G_l, G_l[i])
        
    return G_l

def mle(N, L_b = 1, y_L = -3.8, p0 = 0.25, gamma = 0.5, L = 7):
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
    c_l = sorted(G_l)[int(N0)-1]
    # print("The probability threshold for level 1 is", c_l)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        # print("left at level 1")
        return np.sum(mask) / N, cost
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    dinominator = 1
    
    # level 2, no burn-in
    G_l = sample_new_G(G_l, N, 2, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-4)
    c_l_1 = c_l
    
    c_l = sorted(G_l)[int(N0)-1]
    # print("The probability threshold for level 2 is", c_l)
    
    if c_l <= y_L:
        # print("left at level 2")
        mask = G_l <= y_L
        return p0 * np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    
    G_l_1 = sample_new_G(G_l, N, 2, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-4)
    mask = G_l_1 <= c_l_1
    dinominator *= np.mean(mask)
    # print("The denominator for level 2 is", dinominator)
    
    # level > 2, with burn-in
    N += L_b * N0
    for l in range(3, L):
        G_l = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        # drop the first L_b samples in each Markov chain
        G_l = G_l[int(N0*L_b):]
        c_l_1 = c_l
        
        c_l = sorted(G_l)[int(N0)]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            # print("left at level", l)
            return p0 ** (l-1) * np.sum(mask) / N / dinominator, cost

        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        
        G_l_1 = sample_new_G(G_l, N, l, c_l, gamma = gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        # drop the first L_b samples in each Markov chain
        G_l_1 = G_l_1[int(N0*L_b):]
        mask = G_l_1 <= c_l_1
        dinominator *= np.mean(mask)
        # print("The denominator for level", l, "is", dinominator)
        
    G_l = sample_new_G(G_l, N, L, c_l, gamma = gamma)
    cost += (N - N0) * gamma ** (-2 * L)
    # drop the first L_b samples in each Markov chain
    G_l = G_l[int(N0*L_b):]
    mask = G_l <= y_L
    # print("reach the last level")
    
    return p0 ** (L-1) * np.sum(mask) / N / dinominator, cost

def subset_simulation_sr(N, L=5, gamma=0.5, y_L=-3.8):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    # N: the number of samples per level
    
    # output:
    # p_f: the probability of failure
    # p_f_hat: the probability of failure estimated by the subset simulation
    # p_f_hat_sr: the probability of failure estimated by the subset simulation with the selective refinement
    
    # To compute the sequence of failure thresbolds y_l
    # from genrate_y_l import y_l
    y = [-1.3, -2, -2.8, -3.3, y_L]
    cost_ls = np.zeros(L)
    
    # To generate the samples
    while True:
        G = np.random.normal(0, 1, N)
        kappa = np.random.uniform(-1, 1, N)
        G_l = G + kappa * gamma
        cost_ls[0] = N * gamma ** (-2)
        
        mask = G_l <= y[0]
        # print(mask.sum())
        
        if mask.sum() > 0:
            p_f = mask.mean()
            G_l = G_l[mask][:1]
            break
    
    for l in range(2, L+1):
        while True:
            # print("Level: ", l)
            # print("G_l: ", G_l)
            # time.sleep(0.5)
            for i in range(N-1):
                G_l_new = 0.8 * G_l[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
                kappa_new = np.random.uniform(-1, 1)
                G_l_new += kappa_new * (gamma ** l)
                cost_ls[l-1] += gamma ** (-2 * l)
                
                if G_l_new <= y[l-2]:
                    G_l = np.append(G_l, G_l_new)
                else:
                    G_l = np.append(G_l, G_l[i])
            
            mask = G_l <= y[l-1]
            
            if mask.sum() > 0:
                p_f *= mask.mean()
                G_l = G_l[mask][:1]
                break
            else:
                G_l = G_l[-1:]
                
    cost = np.sum(cost_ls)            
    return p_f, cost

def adaptive_multilevel_subset_simulation(tol, L=5, gamma=0.5, y_L=-3.8):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    
    # output:
    # p_f: the probability of failure
    
    y = [-1.3, -2, -2.8, -3.3, y_L]
    cost_ls = np.zeros(L)
    
    i, err = 0, 10e6
    G_l = np.array([])
    N_l = 0
    while err > tol:
        i += 1
        N_l += 1
        cost_ls[0] += gamma ** (-2)
        
        G = np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        G_l = np.append(G_l, G + kappa * gamma)
        
        mask = G_l <= y[0]
        p_hat = mask.mean()
        
        err = rRMSE(p_hat)
        
    p_f = p_hat
    # print("level: ", 1, "p_hat:", p_hat)
        
    # level 2 to L
    for l in range(1, L):
        G_l = G_l[mask][:1]
        i = 0
        N_l = 1
        err = 10e6
        
        while err > tol:
            G_new = 0.8 * G_l[-1] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
            kappa_new = np.random.uniform(-1, 1)
            G_l_new = G_new + kappa_new * gamma ** (l+1)
            
            if G_l_new <= y[l-1]:
                G_l = np.append(G_l, G_l_new)
            else:
                G_l = np.append(G_l, G_l[-1])
                
            i += 1
            N_l += 1
            cost_ls[l] +=  gamma ** (-2 * (l+1))
                
            mask = G_l <= y[l]
            p_hat = mask.mean()
            
            err = rRMSE(p_f * p_hat)
            
        p_f *= p_hat
        # print("level: ", l+1, "p_hat:", p_hat)
    
    cost = np.sum(cost_ls)  
    return p_f, cost

# Multilevel Estimator with selective refinement
def generate_samples(w, G, N, gamma, c, l, burn_in=0):
    N0 = len(w)
    L_b = burn_in * N0
    cost = 0
    for i in range(N + (burn_in - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        add_term = kappa * gamma
        tol = gamma
        G_new = w_new + add_term
        # cost += 9
        cost += gamma ** (-2)
        for j in range(2, l):
            if tol >= np.abs(G_new - c):
                tol *= gamma
                add_term *= gamma
                G_new = w_new + add_term
                # cost += 3
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
    
    cost = N
    
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

if __name__ == "__main__":
    np.random.seed(10)
    
    # Subset Simulation
    print("Subset Simulation")
    cost_list_ss = []
    err_list_ss = []
    failure_probabilities_ss = []
    for N in [100, 1000, 1600, 3000, 5000]:
        print("N_ss = ", N)
        cost = []
        for i in range(100):
            p_f, c = classical_subset_simulation(N)
            failure_probabilities_ss.append(p_f)
            cost.append(c)
        
        p = np.mean(failure_probabilities_ss)
        print("Failure probabilities: {:.2e}".format(p))
        
        mean_cost = np.mean(cost)
        cost_list_ss.append(mean_cost)
        print("The average cost of SubS is {:.2e}".format(mean_cost))
        
        err = rRMSE(failure_probabilities_ss)
        err_list_ss.append(err)
        print("The relative error of SubS is {:.2e}\n".format(err))
        
    # Multilevel Estimator
    print("MLE")
    err_list_mle = []
    cost_list_mle = []
    for nums in [100, 1000, 1600, 3500, 5000]:
        print("N_mle = ", nums)
        failure_probabilities_mle = []
        costs = []
        for i in range(100):
            p_f, c = mle(nums)
            failure_probabilities_mle.append(p_f)
            costs.append(c)
            
        p = np.mean(failure_probabilities_mle)
        print("The failure probability is {:.2e}".format(p))
        
        cost = np.mean(costs)
        print("The average cost is: {:.2e}".format(cost))
        
        err = rRMSE(failure_probabilities_mle)
        print("The relative error is {:.2e}\n".format(err))
        
        err_list_mle.append(err)
        cost_list_mle.append(cost)
        
    # selective subset simulation
    print("Selective Subset Simulation")
    cost_list_adp = []
    err_list_adp = []
    for N in [1000, 1500, 3000, 10000]:
        print("N_adp = ", N)
        failure_probabilities_adp = np.array([])
        cost_adp = np.array([])
        for _ in range(100):
            p_f, c = subset_simulation_sr(N)
            failure_probabilities_adp = np.append(failure_probabilities_adp, p_f)
            cost_adp = np.append(cost_adp, c)
            
        ave = np.mean(failure_probabilities_adp)
        print("The mean of the failure probability for N = {}: {:.2e}".format(N, ave))
        
        c = np.mean(cost)
        cost_list_adp.append(c)
        print("The average cost: {:.2e}".format(c))
        
        err = rRMSE(failure_probabilities_adp)
        err_list_adp.append(err)
        print("The relative error: {:.2e}\n".format(err))
        
    # adaptive multilevel subset simulation
    print("Adaptive Multilevel Subset Simulation")
    cost_list_adpml = []
    err_list_adpml = []
    for tol in [1, 0.5, 0.1, 0.05, 0.01]:
        print("tolerance: ", tol)
        failure_probabilities_adpml = []
        cost_adpml = []
        for _ in range(100):
            p_f, c = adaptive_multilevel_subset_simulation(tol)
            failure_probabilities_adpml.append(p_f)
            cost_adpml.append(sum(c))
        
        ave = np.mean(failure_probabilities_adpml)
        print("The average probability of failure is: {:.2e}".format(ave))
        
        cost = np.mean(cost_adpml)
        print("The average cost is:", cost)
        cost_list_adpml.append(cost)
        
        err = rRMSE(failure_probabilities_adpml)
        print("The relative error is: {:.2e}\n".format(err))
        err_list_adpml.append(err)
        
    # multi-level estimator with selective refinement
    print("MLE with Selective Refinement")
    cost_list_mle_sr = []
    err_list_mle_sr = []
    for N in [100, 1000, 3000, 4000]:
        print("N_mle_sr: ", N)
        results = [mle_sr(N) for _ in range(100)]
        failure_probabilities, costs = zip(*results)
        
        ave = np.mean(failure_probabilities)
        print("The average probability of failure is: {:.2e}".format(ave))
        
        cost = np.mean(costs)
        cost_list_mle_sr.append(cost)
        print("The mean cost: {:.2e}".format(cost))
        
        err = rRMSE(failure_probabilities)
        err_list_mle_sr.append(err)
        print("The relative error: {:.2e}\n".format(err))
    
    
    # Define the principle line
    
    plt.figure(figsize=(8, 6))
    
    plt.loglog(err_list_ss, cost_list_ss, 'bo-', label='Classical Subset Simulation')
    np.save('err_list_ss.npy', err_list_ss)
    np.save('cost_list_ss.npy', cost_list_ss)
    x = np.linspace(0.01, 0.5, 100)
    plt.loglog(x, 3800000 * x ** (-3), 'b--',  label=r'O($\epsilon^{-2}$)')
    
    # plt.loglog(err_list_adpml, cost_list_adpml, 'yo-', label='Adaptive Multilevel Subset Simulation')
    # np.save('err_list_adpml.npy', err_list_adpml)
    # np.save('cost_list_adpml.npy', cost_list_adpml)
    # plt.loglog(x, 500 * x ** (-2), 'y--',  label=r'O($\epsilon^{-2}$)')
    
    # x = np.linspace(3e-2, 1e-0, 100)
    # plt.loglog(err_list_mle, cost_list_mle, 'ro-', label='MLE')
    # np.save('err_list_mle.npy', err_list_mle)
    # np.save('cost_list_mle.npy', cost_list_mle)
    # plt.loglog(x, 5300000 * x ** (-2), 'r--',  label=r'O($\epsilon^{-2}$)')
    
    x = np.linspace(3e-2, 4e-1, 100)
    plt.loglog(err_list_adp, cost_list_adp, 'go-', label='Selective Subset Simulation')
    np.save('err_list_adp.npy', err_list_adp)
    np.save('cost_list_adp.npy', cost_list_adp)
    plt.loglog(x, 530000 * x ** (-3), 'g--',  label=r'O($\epsilon^{-3}$)')
    
    # x = np.linspace(1e-2, 1, 100)
    # plt.loglog(err_list_mle_sr, cost_list_mle_sr, 'mo-', label='MLE with Selective Refinement')
    # np.save('err_list_mle_sr.npy', err_list_mle_sr)
    # np.save('cost_list_mle_sr.npy', cost_list_mle_sr)
    # plt.plot(x, 100000 * x ** (-1), 'm--', label=r'O($\epsilon^{-1}$)')
    
    
    plt.xlabel('Relative Error')
    plt.ylabel('Cost')
    plt.title('Complexity Results')
    plt.legend()
    plt.show()
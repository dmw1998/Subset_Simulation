import numpy as np
from tqdm import tqdm
import multiprocessing

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    difference = p_hat - 7.23e-05
    expectation = np.mean(difference ** 2)
    return np.sqrt(expectation) / 7.23e-05

def sample_new_G(G_l_in, G_in, N, l, c_l, gamma=0.5):
    N0 = len(G_l_in)
    G_l = np.zeros(N)
    G_l[:N0] = G_l_in
    G = np.zeros(N)
    G[:N0] = G_in
    # r = 0
    # rho = 0.7 + 0.05*l
    rho = 0.8
    
    add = np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, N - N0)
    noise = np.random.uniform(-1, 1, N - N0) * gamma ** (l + 1)
    
    for i in range(N - N0):
        G_new = rho * G[i] + add[i]
        G_l_new = G_new + noise[i]
        
        if G_l_new <= c_l:
            # r += 1
            G_l[i + 1] = G_l_new
            G[i + 1] = G_new
        else:
            G_l[i + 1] = G_l[i]
            G[i + 1] = G[i]
            
    # print("the acceptance rate at level {} is {:.2f}".format(l+1, r / (N - N0)))
    
    return G_l, G

def subset_simulation_y_l(N, y_L=-3.8, gamma=0.5, L=5):
    y = [-1.3, -2, -2.8, -3.3, y_L]
    cost = np.zeros(L)
    
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma
    cost[0] = N * gamma ** (-2)
    
    mask = G_l <= y[0]
    p_f = mask.mean()
    
    for l in range(1, L):
        G_l = G_l[mask][:1]
        G = G[mask][:1]
    
        G_l, G = sample_new_G(G_l, G, N, l, y[l - 1], gamma=gamma)
        cost[l] = (N - 1) * gamma ** (-2 * (l + 1))
        
        mask = G_l <= y[l]
        p_f *= mask.mean()
        
    return p_f, cost

def run_simulation(args):
    N, seed = args
    np.random.seed(seed)
    p_f, cost = subset_simulation_y_l(N)
    return p_f, np.sum(cost)

if __name__ == "__main__":
    error_list = []
    cost_list = []
    np.random.seed(88)

    # for N in [500, 1000, 2000, 5000, 20000, 50000, 100000]:
    #     print("N = ", N)
    #     seeds = np.random.randint(10, 1000, 100)
    #     failure_probabilities = []
    #     costs = []
        
    #     # Use multiprocessing with tqdm progress bar
    #     with multiprocessing.Pool(processes=12) as pool:
    #         # Assign different seeds to each worker
    #         args = [(N, seed) for seed in seeds]
    #         results = list(tqdm(pool.imap(run_simulation, args), total=100))
        
    #     for p_f, cost in results:
    #         failure_probabilities.append(p_f)
    #         costs.append(cost)
            
    #     err = rRMSE(failure_probabilities)
    #     error_list.append(err)
    #     print("The relative RMSE is {:.2e}".format(err))
    #     print("The average computational cost is {:.2e}\n".format(np.mean(costs)))
    #     cost_list.append(np.mean(costs))
        
    #     np.save("SubS_sr_error_list", error_list)
    #     np.save("SubS_sr_cost_list", cost_list)
    
    error_list = np.load("SubS_sr_error_list.npy")[:-1].tolist()
    cost_list = np.load("SubS_sr_cost_list.npy")[:-1].tolist()

    for N in [50000000]:
        print("N = ", N)
        seeds = np.random.randint(100, 1000, 100)
        failure_probabilities = []
        costs = []
        
        # Use multiprocessing with tqdm progress bar
        with multiprocessing.Pool(processes=12) as pool:
            # Assign different seeds to each worker
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, args), total=100))
        
        for p_f, cost in results:
            failure_probabilities.append(p_f)
            costs.append(cost)
            
        err = rRMSE(failure_probabilities)
        error_list.append(err)
        print("The relative RMSE is {:.2e}".format(err))
        print("The average computational cost is {:.2e}\n".format(np.mean(costs)))
        cost_list.append(np.mean(costs))
        
        np.save("SubS_sr_error_list2", error_list)
    #     np.save("SubS_sr_cost_list2", cost_list)
    
    # N = 1000
    # p_f, cost = subset_simulation_y_l(N)
    # print(p_f, cost)
    # print("The relative RMSE is {:.2e}".format(rRMSE(p_f)))
    # print("The average computational cost is {:.2e}".format(np.mean(cost)))
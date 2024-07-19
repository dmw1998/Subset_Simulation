import numpy as np
import time
import matplotlib.pyplot as plt

def k(w):
    return np.random.uniform(-1, 1)

def generate_samples(w, G, N, gamma, c, l, burn_in=0):
    N0 = len(w)
    L_b = burn_in * N0
    for i in range(N + (burn_in - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = k(w_new)
        add_term = kappa * gamma
        tol = gamma
        G_new = w_new + add_term
        for j in range(2, l):
            if tol >= np.abs(G_new - c):
                tol = gamma ** j
                add_term *= gamma
                G_new = w_new + add_term
            else:
                break
            
        if G_new <= c:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    return w[L_b:], G[L_b:]

def mle_sr(gamma, y, p_0, N, L, burn_in):
    
    N0 = int(p_0 * N)
    
    l = 1
    w = np.random.normal(0, 1, N)
    kappa = np.array([k(g) for g in w])
    G = w + kappa * gamma
    
    c_1 = np.sort(G)[N0-1]
    
    cost = N
    
    # For l = 2, no burn-in
    mask = G <= c_1
    w = w[mask][:N0]
    G = G[mask][:N0]
    
    w, G = generate_samples(w, G, N, gamma, c_1, 2)
    
    cost += N - N0
    
    c_2 = np.sort(G)[N0-1]
    _, G_2 = generate_samples(w, G, N, gamma, c_2, 2)
    
    cost += N - N0
    
    mask = G_2 <= c_1
    denominator = np.mean(mask)
    
    # For l > 2, burn-in
    c_l = c_2
    for l in range(3,L):
        c_l_1 = c_l
        w, G = generate_samples(w, G, N, gamma, c_l, l, burn_in)
        cost += N + (burn_in - 1) * N0
        c_l = np.sort(G)[N0-1]
        
        if c_l <= y:
            mask = G <= y
            failure_probability = p_0 ** (l-1) * np.mean(mask) / denominator
            return failure_probability, cost
        
        mask = G <= c_l
        w = w[mask][:N0]
        G = G[mask][:N0]
        
        _, G_l = generate_samples(w, G, N, gamma, c_l, l, burn_in)
        cost += N + (burn_in - 1) * N0
        
        mask = G_l <= c_l_1
        denominator *= np.mean(mask)
        
    # For l = L
    c_L_1 = c_l
    w, G = generate_samples(w, G, N, gamma, c_l, L, burn_in)
    cost += N + (burn_in - 1) * N0
    
    mask = G <= y
    w = w[mask]
    G = G[mask]
    p_L = np.mean(mask)
    
    _, G_L = generate_samples(w, G, N, gamma, y, L, burn_in)
    cost += N + (burn_in - 1) * N0
    
    mask = G_L < c_L_1
    denominator *= np.mean(mask)
    
    failure_probability = p_0 ** (L-1) * p_L / denominator
    
    return failure_probability, cost

if __name__ == "__main__":
    L = 5
    gamma = 0.5
    y = -3.8
    p_0 = 0.1
    N = 50000
    burn_in = 10
    
    np.random.seed(2)
    # start = time.time()
    # p_f, cost = mle_sr(gamma, y, p_0, N, L, burn_in)
    # print("Time: ", time.time() - start)
    # print("Failure probability: ", p_f)
    # print("Cost: ", cost)
    # print("Relative error: ", 1/7.23e-5*np.sqrt((p_f - 7.23e-05)**2))
    for N in [1000, 5000, 10000, 50000]:
        print("N: ", N)
        start = time.time()
        results = [mle_sr(gamma, y, p_0, N, L, burn_in) for _ in range(100)]
        failure_probabilities, costs = zip(*results)
        print("Time: ", time.time() - start)
        # print("The mean of the failure probability: ", np.mean(failure_probabilities))
        print("The mean cost: ", np.mean(costs))
        rRMSE = 1 / 7.23e-05 * np.sqrt(np.mean((np.array(failure_probabilities) - 7.23e-05) ** 2))
        print("rRMSE: ", rRMSE)
        print("")
    
    # from confidence_interval import bootstrap_confidence_interval
    
    # confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities)

    # print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    # print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    # # Plot the cdf of the failure probabilities
    # p_f = np.sort(failure_probabilities)
    # cdf = np.arange(1, len(p_f) + 1) / len(p_f)
    
    # # Plot the empirical CDF
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
    # plt.step(p_f, cdf, where='post')
    # plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    # plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    # plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    # plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='90% Confidence Interval')
    # plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    # plt.xlabel('Probability')
    # plt.ylabel('Empirical CDF')
    # plt.title('Empirical CDF of Probabilities')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # Plot RMSE against cost
    # plt.figure(figsize=(8, 6))
    # plt.scatter(costs, rmses, alpha=0.5)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.xlabel('Cost')
    # plt.ylabel('RMSE')
    # plt.title('RMSE against Cost')
    # plt.grid(True)
    # plt.show()

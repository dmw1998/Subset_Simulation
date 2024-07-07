import numpy as np
import matplotlib.pyplot as plt
import time

def test_4():
    # Parameters
    gamma = 0.5
    y = -3.8        # Critical value
    p_0 = 0.1       # Conditional probability for each subset
    N = 1000         # Number of samples per level
    L_b = 5        # Burn-in length
    L = 5           # Number of levels
    N0 = int(p_0 * N)
    
    # Sample N i.i.d. standard normal random variables w
    w = np.random.normal(0, 1, N)
    
    # Compute G_1 = w + kappa * gamma for the first level
    kappa = np.random.uniform(-1, 1, N)
    G = w + kappa * gamma
    
    # Determine the threshold c_1
    c_1 = np.sort(G)[N0]
    
    # For l = 2, n = 0, no burn-in
    l = 2
    
    # N0 failure points in F_1 are
    mask = G < c_1
    w = w[mask]
    G = G[mask]
    
    # Generate the samples in F_1
    for i in range(N - N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - c_1):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new < c_1:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
    
    # determine the threshold c_2
    c_2 = np.sort(G)[N0]
    
    # N0 failure points in F_2 are
    mask = G < c_2
    w = w[mask]
    G = G[mask]
    
    # Generate the samples in F_2
    for i in range(N - N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < 2 and tol >= np.abs(G_new - c_2):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= c_2:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    # Evaluate the failure probability F_1|F_2
    mask = G < c_1
    denominator = np.sum(mask) / N
    
    # l = 3, n = L_b, burn-in
    l = 3
    
    # N0 failure points in F_2 are
    w = w[:N0]
    G = G[:N0]
    
    # Generate the samples in F_2
    for i in range(N + (L_b - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - c_1):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= c_2:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
    
    # Burn-in
    w = w[L_b * N0:]
    G = G[L_b * N0:]
            
    # Determine the threshold c_3
    c_3 = np.sort(G)[N0]
    
    # N0 failure points in F_3 are
    mask = G < c_3
    w_3 = w[mask]
    G_3 = G[mask]
    w = w_3
    G = G_3
    
    # Generate the samples in F_3
    for i in range(N + (L_b - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - c_3):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= c_3:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    # Burn-in
    w = w[L_b * N0:]
    G = G[L_b * N0:]
    
    # Evaluate the failure probability F_2|F_3
    mask = G < c_2
    denominator *= np.sum(mask) / N
    
    # l = 4, n = L_b, burn-in
    l = 4
    
    # N0 failure points in F_3 are
    w = w_3
    G = G_3
    
    # Generate the samples in F_3
    for i in range(N + (L_b - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - c_3):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= c_3:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    # Burn-in
    w = w[L_b * N0:]
    G = G[L_b * N0:]
    
    # Determine the threshold c_4
    c_4 = np.sort(G)[N0]
    
    # N0 failure points in F_4 are
    mask = G < c_4
    w_4 = w[mask]
    G_4 = G[mask]
    w = w_4
    G = G_4
    
    # Generate the samples in F_4
    for i in range(N + (L_b - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - c_4):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= c_4:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    # Burn-in
    w = w[L_b * N0:]
    G = G[L_b * N0:]
    
    # Evaluate the failure probability F_3|F_4
    mask = G < c_3
    denominator *= np.sum(mask) / N
    
    # l = 5, n = L_b, burn-in
    l = 5
    
    # N0 failure points in F_4 are
    w = w_4
    G = G_4
    
    # Generate the samples in F_4
    for i in range(N + (L_b - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - c_4):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= c_4:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    # Burn-in
    w = w[L_b * N0:]
    G = G[L_b * N0:]
    
    # Evaluate the failure probability F_5|F_4
    mask = G < y
    w_L = w[mask]
    G_L = G[mask]
    p_L = np.sum(mask) / N
    
    # N0 failure points in F_5 are
    w = w_L
    G = G_L
    
    # Gererate the samples in F_5
    for i in range(N + (L_b - 1) * N0):
        w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        j = 1
        tol = gamma
        G_new = w_new + kappa * gamma
        while j < l and tol >= np.abs(G_new - y):
            j += 1
            tol = gamma ** j
            G_new = w_new + kappa * gamma ** j
        
        if G_new <= y:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            w = np.append(w, w[i])
            G = np.append(G, G[i])
            
    #Burn-in
    w = w[L_b * N0:]
    G = G[L_b * N0:]
    
    # Evaluate the failure probability F_4|F_5
    mask = G < c_4
    denominator *= np.sum(mask) / N
    
    failure_probability = p_0 ** 4 * p_L / denominator
    
    return failure_probability

if __name__ == "__main__":
    np.random.seed(1)
    start = time.time()
    failure_probabilities = [test_4() for _ in range(1000)]
    print("Time: ", time.time() - start)
    print("The mean of the failure probability: ", np.mean(failure_probabilities))
    
    from confidence_interval import bootstrap_confidence_interval
    
    # Calculate 95% confidence interval using bootstrap method
    confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.step(p_f, cdf, where='post')
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='90% Confidence Interval')
    plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()
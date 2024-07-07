import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 0.5
y = -3.8        # Failure threshold
p_0 = 0.1       # Conditional probability for each subset
N = 100        # Number of samples per level

# Pointwise approximation G_l
def G_l(w, l):
    kappa = np.random.uniform(-1, 1)
    
    return w + kappa * (gamma ** l)

# Selective refinement
def selective_refinement(w, l, y, gamma):
    G_j = G_l(w, 0)
    j = 0
    tol = 1
    
    while j < l and tol >= np.abs(G_j - y):
        j += 1
        tol = gamma ** j
        G_j = G_l(w, j)
        
    return G_j

# Subset simulation with selective refinement
def subset_simulation_with_selective_refinement(N, p_0, y, gamma):
    
    N0 = int(p_0 * N)-1
    
    w = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G = w + kappa * gamma
    
    c_l = np.sort(G)[N0]
    # print("c_ 1", c_l)
    
    l = 1
    # failure_probability = 1
    
    while c_l > y and l < 4:
        mask = G <= c_l
        # print("Number of samples in level", l, ":", np.sum(mask))
        # failure_probability *= np.sum(mask) / N
        w = w[mask][:N0]
        G = G[mask][:N0]
        
        l += 1
        
        for i in range(N - N0):
            w_new = 0.8 * w[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
            # w_new = np.random.normal(0, 1)
            
            G_new = selective_refinement(w_new, l, c_l, gamma)
            
            if G_new <= c_l:
                w = np.append(w, w_new)
                G = np.append(G, G_new)
            else:
                w = np.append(w, w[i])
                G = np.append(G, G[i])
            # w = np.append(w, w_new)
            # G = np.append(G, G_new)
            
        c_l = np.sort(G)[N0]
        # print("c_",l, ":", c_l)
    
    print("Number of samples in the last level", l+1, ":", np.sum(G <= y))        
    failure_probability = (p_0 ** l) * (np.sum(G <= y) / N)
    # failure_probability *= np.sum(G <= y) / N
    
    return failure_probability

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    n = len(data)
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, n), replace=True)
    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
    
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return lower_bound, upper_bound

np.random.seed(0)
failure_probabilities = [subset_simulation_with_selective_refinement(N, p_0, y, gamma) for _ in range(1000)]
# print(failure_probabilities)

confidence_interval = bootstrap_confidence_interval(failure_probabilities)

print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))

p_f = sorted(failure_probabilities)
cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

# Plot the empirical CDF
plt.figure(figsize=(8, 6))
plt.xscale("log")
plt.xlim(5e-6, 5e-4)
plt.step(p_f, cdf, where='post')
plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
plt.xlabel('Probability')
plt.ylabel('Empirical CDF')
plt.title('Empirical CDF of Probabilities')
plt.legend()
plt.grid(True)
plt.show()
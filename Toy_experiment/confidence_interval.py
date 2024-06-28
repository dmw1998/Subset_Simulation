import numpy as np

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000):
    n = len(data)
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, n), replace=True)
    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
    
    lower_percentile = (1.0 - 0.95) / 2.0 * 100
    upper_percentile = (1.0 + 0.95) / 2.0 * 100
    
    lower_percentile_90 = (1.0 - 0.90) / 2.0 * 100
    upper_percentile_90 = (1.0 + 0.90) / 2.0 * 100
    
    lower_bound_95 = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound_95 = np.percentile(bootstrap_estimates, upper_percentile)
    
    lower_bound_90 = np.percentile(bootstrap_estimates, lower_percentile_90)
    upper_bound_90 = np.percentile(bootstrap_estimates, upper_percentile_90)
    
    return (lower_bound_95, upper_bound_95), (lower_bound_90, upper_bound_90)
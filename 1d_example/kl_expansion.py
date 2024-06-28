# Approximate a lognormal random field using the Karhunen-Loeve expansion
# Since the random field is one dimensional, we can use the analytical solution

import numpy as np
import matplotlib.pyplot as plt

def kl_expan(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    # Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
    # def eigenvalue(m):
    #     return 0.02 / np.pi ** 2 / (m + 0.5) ** 2
    
    # def eigenfunction(m, x):
    #     return np.sqrt(2) * (np.sin((m + 0.5) * np.pi * x) + np.cos((m + 0.5) * np.pi * x))
    
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m*np.pi
        return 2*beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m*np.pi
        A = np.sqrt(2 * w**2 / (2*beta + w**2 + beta**2))
        B = np.sqrt(2 * beta**2 / (2*beta + w**2 + beta**2))
        return A*np.cos(w*x) + B*np.sin(w*x)
    
    # p = 0
    # for m in range(M):
    #     p += np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1,x)
    #     plt.plot(x, p)
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * thetas[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x


def kl_expan_1(thetas):
    """
    Compute the KL expansion for the log-normal random field.

    Args:
    thetas (numpy.ndarray): Array of length M, the KL expansion coefficients.

    Returns:
    a_x (numpy.ndarray): Array of length n, the random field values at the spatial points.
    """
    
    M = len(thetas)  # Number of KL terms
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)  # Spatial grid points
    
    # Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m * np.pi
        return 2 * beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m * np.pi
        return np.sqrt(2) * np.sin(w * x)
    
    # Compute the mean and standard deviation for log-normal distribution
    mu_a = 1
    sigma_a = 0.1
    mu = np.log(mu_a**2 / np.sqrt(mu_a**2 + sigma_a**2))
    sigma = np.sqrt(np.log(1 + (sigma_a**2 / mu_a**2)))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(
        np.sqrt(eigenvalue(m + 1)) * eigenfunction(m + 1, x) * thetas[m] for m in range(M)
    )

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x


def kl_expansion_corr(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    # Define the covariance function based on the correlation length
    def covariance(x1, x2):
        return np.exp(-np.abs(x1 - x2) / 0.01)
    
    # Create the covariance matrix
    cov_matrix = np.array([[covariance(xi, xj) for xj in x] for xi in x])

    # Eigenvalue decomposition to find eigenvalues and eigenfunctions
    eigenvalues, eigenfunctions = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenfunctions in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][:M]
    eigenfunctions = eigenfunctions[:, idx][:, :M]
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))

    # Construct log(a(x)) using the KL expansion
    log_a_x = mu + sigma * np.sum([np.sqrt(eigenvalues[n]) * eigenfunctions[:, n] * thetas[n] for n in range(M)], axis=0)

    # Construct a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

if __name__ == "__main__":    
    # Define the correlation length
    l_c = 0.01
    
    M = 150
    
    # Define the KL coefficients
    np.random.seed(0)
    thetas = np.random.normal(0, 1, M)
    # thetas = np.ones(M)
    
    # Compute the random field
    a_x = kl_expan(thetas)
    a_x_corr = kl_expansion_corr(thetas)
    x = np.linspace(0, 1, 1000)
    
    # Plot the random field
    plt.plot(x,a_x)
    plt.plot(x,a_x_corr)
    plt.show()
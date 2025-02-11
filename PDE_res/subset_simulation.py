# Subset simulation with fully refinement

from fenics import *
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


def sampling_one_new_theta(G, theta_1, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # G: the approximated IoQ
    # theta_1: current state
    # c_l: threshold value
    # u_max: critial value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_new: new state
    
    M = len(theta_1)
    # theta_c = np.zeros_like(theta_1)
    # for i in range(M):
    #     theta = theta_1[i]
        
    #     theta_tilde = gamma * theta + np.sqrt(1 - gamma**2) * np.random.normal()
        
    #     # Compute the ratio r = f(theta_tilde) / f(theta), 
    #     # where f is the pdf of the Gaussian distribution
    #     # f(x) = exp(-0.5 * x^2) / sqrt(2 * pi)
    #     # r = exp(-0.5 * (theta_tilde**2 - theta**2))
    #     r = exp(0.5 * (theta**2 - theta_tilde**2))
        
    #     if np.random.rand() < min(1, r):
    #         theta_c[i] = theta_tilde
    #     else:
    #         theta_c[i] = theta
    
    theta_c = gamma * theta_1 + np.sqrt(1 - gamma**2) * np.random.randn(M)
        
    # Solve the PDE for required mesh size
    u_h = IoQ(kl_expan(theta_c), n_grid)
    
    # Acceptance condition
    if u_max - u_h <= c_l:
        # If theta_c in F_{l-1}, then accept theta_c
        theta_new = theta_c
        G_new = (u_max - u_h)
    else:
        # If theta_c not in F_{l-1}, then reject theta_c
        theta_new = theta_1
        G_new = G[-1]
    
    return G_new, theta_new

def sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # c_l: threshold value
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N - N0:
        theta = theta_ls[i]
        G_new, theta_new = sampling_one_new_theta(G, theta, c_l, u_max, n_grid, gamma = gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
    
    return G, theta_ls

def sampling_theta_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # L_b: burn-in length
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # c_l: threshold value
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N + (L_b - 1) * N0:
        theta = theta_ls[i]
        G_new, theta_new = sampling_one_new_theta(G, theta, c_l, u_max, n_grid, gamma = gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
        
    G = G[N0 * L_b:]
    theta_ls = theta_ls[N0 * L_b:]
    
    return G, theta_ls

def IoQ(a_x, n_grid):
    # input:
    # a_x: the random field (coefficients)
    # n_grid: number of mesh points
    
    # output:
    # u_h(1): approximated IoQ
    
    # Set PETSc options to suppress solver output
    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    
    # Create the mesh and define function space
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    # Define the random field a(x) on the FEniCS mesh
    a = Function(V)
    a_values = np.interp(mesh.coordinates().flatten(), np.linspace(0, 1, len(a_x)), a_x)
    a.vector()[:] = a_values
    
    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    bc = DirichletBC(V, u0, boundary)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    
    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Compute solution
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)  # Suppress the FEniCS log messages
    solve(a_form == L, u_h, bc)
    
    return u_h(1)

def compute_cl(G, thetas_ls, N, p0, l, L):
    # input:
    # G: approximated IoQ
    # thetas_ls: list of thetas
    # N: number of samples
    # p0: failure probability
    # l: current level
    # L: finest level
    
    # output:
    # G, thetas_ls: updated G and thetas_ls
    # c_l: failure level
    
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_theta_ls = [thetas_ls[i] for i in sorted_indices]
    
    N0 = int(p0 * N)
    c_l = sorted_G[N0-1]
    if c_l < 0 or l == L:
        # When we reach the finest level
        G = [g for g in G if g < 0]
    else:
        G = sorted_G[:N0]
        thetas_ls = sorted_theta_ls[:N0]
    
    return G, thetas_ls, c_l

def subset_simulation(args, M=150, p0=0.1, u_max=0.535, n_grid=512, gamma = 0.8, L = 5):
    # input:
    # N: number of required samples
    # M: number of terms in the KL expansion
    # p0: failure probability
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    # L: number of levels
    
    # output:
    # p_f: failure probability
    
    N, seed = args
    np.random.seed(seed)
    
    # Initialize the list of the approximated IoQ
    G = []
    
    # Initialize the list of the initial states
    theta_ls = []
    
    p_f = p0
    
    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        theta_ls.append(thetas)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_max - u_1
        G.append(g)
    
    # Compute the threshold value
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 0, L)
    
    # print("Level: 0", "Threshold value: ", c_l)
    
    if c_l < 0:
        return len(G) / N
    
    # For l = 2, ..., L
    for l in range(1, L+1):
        # Generate N - N0 samples for each level
        G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma)
        
        # Compute the threshold value
        G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, l, L)
        
        # print("Level:", l, "Threshold value: ", c_l)
        
        if c_l < 0:
            break
        
        p_f *= p0
    
    return p_f * len(G) / N

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    n = len(data)
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, n), replace=True)
    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
    
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return lower_bound, upper_bound

if __name__ == "__main__":
    # # Define the number of simulations
    # num_simulations = 500
    
    # # Define the number of samples
    # N = 1000
    
    # # Define the number of terms in the KL expansion
    # M = 150
    
    # # Define the failure probability
    # p0 = 0.1
    
    # # Define the upper bound of the solution
    # u_max = 0.535
    
    # # Define the number of mesh points
    # n_grid = 512
    
    # # Define the correlation parameter
    # gamma = 0.8
    
    # # Define the number of levels
    # L = 5
    
    # np.random.seed(1)
    # # Compute the probability of failure
    # p_f = subset_simulation(N, M, p0, u_max, n_grid, gamma, L)
    # print("The probability of failure is: {:.2e}".format(p_f))
    
    # # np.random.seed(0)
    # # runs = 10
    # # p_f = np.zeros(runs)
    # # for i in range(0, runs):
    # #     # print("Seed: ", i)
    # #     # np.random.seed(i)
    # #     # Compute the probability of failure
    # #     p_f[i] = subset_simulation(N, M, p0, u_max, n_grid, gamma, L)
    # #     print("The probability of failure is: {:.2e}".format(p_f[i]))
        
    # #     # print("")
    
    # # # save p_f
    # # # np.save("p_f.npy", p_f)
    # # print("The mean of the probability of failure is: {:.2e}".format(np.mean(p_f)))
    
    # failure_probabilities = [subset_simulation(N, M, p0, u_max, n_grid, gamma, L) for _ in range(num_simulations)]
    # # print("Failure probabilities:", failure_probabilities[0:10])

    # # Calculate 95% confidence interval using bootstrap method
    # confidence_interval = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=100, confidence_level=0.95)

    # print("95% confidence interval for failure probability:", confidence_interval)
    
    # p_f = sorted(failure_probabilities)
    # cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # # Step 3: Plot the empirical CDF
    # plt.figure(figsize=(8, 6))
    # plt.xscale("log")
    # plt.xlim(1e-5, 1e-3)
    # plt.step(p_f, cdf, where='post')
    # plt.xlabel('Probability')
    # plt.ylabel('Empirical CDF')
    # plt.title('Empirical CDF of Probabilities')
    # plt.grid(True)
    # plt.show()
    
    def rRMSE(p_f):
        return np.sqrt(np.mean((p_f - 1.6e-04) ** 2)) / 1.6e-04
    
    from tqdm import tqdm
    import multiprocessing as mp
    
    np.random.seed(26)
    error_list = []
    cost_list = []
    for N in [500, 1000, 2000, 4000, 6000]:
        failure_probabilities = []
        cost = []
        with mp.Pool(processes=12) as pool:
            for p_f in tqdm(pool.imap(subset_simulation, [(N, i) for i in range(100)]), total=100, desc=f"N = {N}"):
                failure_probabilities.append(p_f)
                
            cost = (N + 4 * (N - N*0.1)) * 512
            cost_list.append(cost)
            failure_probabilities = np.array(failure_probabilities)
            error = rRMSE(failure_probabilities)
            
            print(f"Failure probability: {p_f:.2e}")
            print(f"Error: {error:.2e}")
            print(f"Cost: {cost:.2e}\n")
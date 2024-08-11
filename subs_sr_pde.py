import numpy as np
from fenics import *

# Apply the subset simulation method with selective refinement strategy to the PDE model.

# Generate the samples for the Gaussian random field.
# We need to write a new method. Combine the subset simulation and the selective refinement strategy.

# Computing the coefficient for the pde
def kl_expan(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m*np.pi
        return 2*beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m*np.pi
        A = np.sqrt(2 * w**2 / (2*beta + w**2 + beta**2))
        B = np.sqrt(2 * beta**2 / (2*beta + w**2 + beta**2))
        return A*np.cos(w*x) + B*np.sin(w*x)
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * thetas[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

# Solving the 1d poisson equation with the finite element method.
def solving_pde(theta, l):
    # input:
    # theta: the parameter of the Gaussian random field
    # l: the level -- related to mesh size

    # output:
    # u_h(1): the QoI on level l

    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    
    n_grid = 64 * 2 ** l
    a_x = kl_expan(theta)
    
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
    
# Sampling the parameter theta follows the Gaussian distribution
# Modified MH algorithm
def generate_samples(theta, G, N, gamma, y, l, burn_in=0):
    # input:
    # theta: the initial parameter
    # G: the initial QoI
    # N: the number of samples
    # gamma: the step size
    # y: the threshold
    # l: the current level
    # burn_in: the burn-in period

    # output:
    # theta: the generated samples
    # G: the corresponding QoI
    
    N0 = len(theta)
    L_b = burn_in * N0
    for i in range(N + (burn_in - 1) * N0):
        theta_new = 0.8 * theta[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        G_new = solving_pde(theta_new, 0)
        tol = 1
        for j in range(1, l):
            if tol >= np.abs(G_new - y):
                tol *= gamma
                G_new = solving_pde(theta_new, j)
            else:
                break
            
        if G_new <= y:
            theta = np.append(theta, theta_new)
            G = np.append(G, G_new)
        else:
            theta = np.append(theta, theta[i])
            G = np.append(G, G[i])
            
    return theta[L_b:], G[L_b:]

# Apply the subset simulation method with selective refinement strategy to the PDE model.
def mle_sr(gamma, y, p_0, N, L, burn_in):
    
    N0 = int(p_0 * N)
    
    l = 0
    thetas = np.random.normal(0, 1, N)
    G = np.array([solving_pde(thetas[i], l) for i in range(N)])
    
    c_1 = np.sort(G)[N0-1]
    
    cost = N      # The times of solving the PDE
    
    # For l = 2, no burn-in
    mask = G <= c_1
    thetas = thetas[mask][:N0]
    G = G[mask][:N0]
    
    thetas, G = generate_samples(thetas, G, N, gamma, c_1, 2)
    
    cost += N - N0
    
    c_2 = np.sort(G)[N0-1]
    _, G_2 = generate_samples(thetas, G, N, gamma, c_2, 2)
    
    cost += N - N0
    
    mask = G_2 <= c_1
    denominator = np.mean(mask)
    
    # For l > 2, burn-in
    c_l = c_2
    for l in range(3,L):
        c_l_1 = c_l
        thetas, G = generate_samples(thetas, G, N, gamma, c_l, l, burn_in)
        cost += N + (burn_in - 1) * N0
        c_l = np.sort(G)[N0-1]
        
        if c_l <= y:
            mask = G <= y
            failure_probability = p_0 ** (l-1) * np.mean(mask) / denominator
            return failure_probability, cost
        
        mask = G <= c_l
        thetas = thetas[mask][:N0]
        G = G[mask][:N0]
        
        _, G_l = generate_samples(thetas, G, N, gamma, c_l, l, burn_in)
        cost += N + (burn_in - 1) * N0
        
        mask = G_l <= c_l_1
        denominator *= np.mean(mask)
        
    # For l = L
    c_L_1 = c_l
    thetas, G = generate_samples(thetas, G, N, gamma, c_l, L, burn_in)
    cost += N + (burn_in - 1) * N0
    
    mask = G <= y
    thetas = thetas[mask]
    G = G[mask]
    p_L = np.mean(mask)
    
    _, G_L = generate_samples(thetas, G, N, gamma, y, L, burn_in)
    cost += N + (burn_in - 1) * N0
    
    mask = G_L < c_L_1
    denominator *= np.mean(mask)
    
    failure_probability = p_0 ** (L-1) * p_L / denominator
    
    return failure_probability, cost
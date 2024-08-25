import numpy as np
from fenics import *

def kl_expan(theta):
    
    M = len(theta)
    
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
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * theta[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

def solving_pde(theta, l):
    
    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    
    n_grid = int(64 * 2 ** (l-1))
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
    
    return 0.535 - u_h(1)

def compute_cl(G, thetas, N, p_0, l, L):
    
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_thetas = [thetas[i] for i in sorted_indices]
    
    N_0 = int(p_0 * N)
    c_l = sorted_G[N_0-1]
    if c_l < 0 or l == L:
        G = [g for g in G if g < 0]
    else:
        G = sorted_G[:N_0]
        thetas = sorted_thetas[:N_0]
    
    return G, thetas, c_l

def sampling(N, G, thetas, c_l, l, gamma, burn_in=0):
    
    M = len(thetas[0])
    N0 = len(G)
    L_b = burn_in * N0
    cost = 0
    
    for i in range(N + (burn_in - 1) * N0):
        theta_new = 0.8 * thetas[i] + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1, M)
        G_new = solving_pde(theta_new, 1)
        cost += 1
        tol = gamma
        for j in range(2, l):
            if tol >= np.abs(G_new - c_l):
                tol = gamma ** j
                G_new = solving_pde(theta_new, j)
                cost += j
            else:
                break
        
        if G_new <= c_l:
            thetas = np.append(thetas, theta_new)
            G = np.append(G, G_new)
        else:
            thetas = np.append(thetas, thetas[i])
            G = np.append(G, G[i])
            
    return thetas[L_b:], G[L_b:], cost

def mle_sr(gamma, y, p_0, N, L, burn_in):
    
    l = 1
    thetas = [np.random.normal(0, 1, 150) for _ in range(N)]
    G = [solving_pde(theta, 0) for theta in thetas]
    cost = N * l
    
    G, thetas, c_1 = compute_cl(G, thetas, N, p_0, l, L)
    print('c_1 = ', c_1)
    
    l = 2
    thetas, G, add_cost = sampling(N, G, thetas, c_1, l, gamma, burn_in=0)
    cost += add_cost
    
    G, thetas, c_2 = compute_cl(G, thetas, N, p_0, l, L)
    print('c_2 = ', c_2)
    
    _, G_2, add_cost = sampling(N, G, thetas, c_2, l, gamma, burn_in=0)
    cost += add_cost
    
    mask = np.array(G_2) <= c_1
    denominator = np.mean(mask)
    
    # For l > 2, burn-in
    c_l = c_2
    for l in range(3, L):
        c_l_1 = c_l
        thetas, G, add_cost = sampling(N, G, thetas, c_l, l, gamma, burn_in)
        cost += add_cost
        G, thetas, c_l = compute_cl(G, thetas, N, p_0, l, L)
        print('c_', l, ' = ', c_l)
        
        if c_l < 0:
            mask = np.array(G) <= 0
            failure_probability = p_0 ** (l-1) * np.mean(mask) / denominator
            return failure_probability, cost
        
        _, G_l, add_cost = sampling(N, G, thetas, c_l, l, gamma, burn_in)
        cost += add_cost
        
        mask = np.array(G_l) <= c_l_1
        denominator *= np.mean(mask)
    
    # l = L
    c_L_1 = c_l
    thetas, G, add_cost = sampling(N, G, thetas, c_l, L, gamma, burn_in)
    cost += add_cost
    
    G, thetas, _ = compute_cl(G, thetas, N, p_0, L, L)
    p_L = len(G) / N
    
    _, G_L, add_cost = sampling(N, G, thetas, y, L, gamma, burn_in)
    cost += add_cost
    
    mask = np.array(G_L) <= c_L_1
    denominator *= np.mean(mask)
    
    failure_probability = p_0 ** (L-1) * p_L / denominator
    
    return failure_probability, cost

if __name__ == "__main__":
    p_0 = 0.25
    N = 1000
    gamma = 0.8
    y = 0.535
    L = 3
    burn_in = 10
    
    np.random.seed(0)
    p_f, cost = mle_sr(gamma, y, p_0, N, L, burn_in)
    print("The probability of failure is: {:.2e}".format(p_f))
    print("The cost is: ", cost)
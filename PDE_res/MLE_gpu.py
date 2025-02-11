from fenics import *
import numpy as np
import os
import cupy as cp

# Set PETSc options to suppress solver output (and write to file)
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    difference = p_hat - 7.23e-05
    expaction = np.mean(difference ** 2)
    return np.sqrt(expaction) / 7.23e-05


# Precompute eigenvalues and eigenfunctions
M = 150  # Example value
x = cp.linspace(0, 1, 1000)
beta = 1 / 0.01
eigenvalues = cp.array([2 * beta / ((m * cp.pi)**2 + beta**2) for m in range(1, M + 1)])
eigenfunctions = cp.array([[cp.sqrt(2 * (m * cp.pi)**2 / (2 * beta + (m * cp.pi)**2 + beta**2)) * cp.cos(m * cp.pi * xi) +
                             cp.sqrt(2 * beta**2 / (2 * beta + (m * cp.pi)**2 + beta**2)) * cp.sin(m * cp.pi * xi)
                             for xi in x] for m in range(1, M + 1)])

def kl_expan(theta):
    mu = -0.5 * cp.log(1.01)
    sigma = cp.sqrt(cp.log(1.01))
    log_a_x = mu + sigma * cp.sum(cp.sqrt(eigenvalues)[:, cp.newaxis] * eigenfunctions * theta[:, cp.newaxis], axis=0)
    return cp.exp(log_a_x)


def IoQ(a_x, n_grid):
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)

    a = Function(V)
    a_values = cp.interp(mesh.coordinates().flatten(), cp.linspace(0, 1, len(a_x)), a_x)
    a.vector()[:] = a_values

    u0 = Constant(0.0)

    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    bc = DirichletBC(V, u0, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)

    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx

    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L, u_h, bc, solver_parameters={"linear_solver": "mumps"}) # Example solver
    return u_h(1)


def mh_sampling(N, G, thetas, c_l, u_max, n_grid, l, M, gamma=0.8):
    N0 = len(G)
    sample_numbers = np.zeros(7)
    add_term = cp.sqrt(1 - gamma**2) * cp.random.normal(0, 1, (N-N0, M))
    
    for i in range(N - N0):
        tol = 0.5    # Pre-computed value
        
        seed = thetas[i:i + 1]
        candidate = gamma * seed + add_term[i:i+1]
        G_new = u_max - IoQ(kl_expan(candidate[0]), n_grid)
        sample_numbers[0] += 1
        
        for j in range(2, l+1):
            if tol >= cp.abs(G_new - c_l):
                tol *= 0.5
                n_grid *= 2
                G_new = u_max - IoQ(kl_expan(candidate[0]), n_grid)
                sample_numbers[j-1] += 1
            else:
                break
        
        if G_new <= c_l:
            G = cp.append(G, G_new)
            thetas = cp.append(thetas, candidate, axis=0)
        else:
            G = cp.append(G, G[i])
            thetas = cp.append(thetas, seed, axis=0)
            
    return cp.asnumpy(G), cp.asnumpy(thetas), sample_numbers


def mle(N, seed_offset):
    cp.random.seed(26 + seed_offset)  # Unique seed for each worker
    
    p0 = 0.25
    M = 150
    L_b = 10
    u_max = 0.535
    n_grid = 8
    L = 7
    
    N0 = int(N * p0)
    P = 100 * p0
    
    # Generate initial samples
    G = cp.zeros(N)
    sample_numbers = np.zeros(L)
    
    theta_ls = cp.random.normal(0, 1, (N, M))
    for i in range(N):
        u_1 = IoQ(kl_expan(theta_ls[i]), n_grid)
        g = u_max - u_1
        G[i] = g
        
    sample_numbers[0] = N
    
    # Determine the threshold value c_l
    c_l = cp.percentile(G, P)
    print('c_1 = ', c_l)
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return len(G) / N, sample_numbers
    
    denominator = 1
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0][:]
    
    # l = 2 < 3, set L_b = 0    
    # Sampling without burn-in
    c_l_1 = c_l    # c_{l-1}
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, 2, M, gamma = 0.8)
    sample_numbers += add_num
    print("len(G): ", len(G)) # Debug
    
    c_l = cp.percentile(G, P)
    print('c_2 = ', c_l)
    
    G_, _, add_num = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, 2, M, gamma = 0.8)
    sample_numbers += add_num
    
    denominator *= cp.mean(G_ <= c_l_1)
    # print('denominator =', denominator)
    
    if c_l < 0:
        return p0 * cp.mean(G <= 0) / denominator, sample_numbers
    
    # l > 2, set L_b = L_b
    # Sampling with burn-in
    N += L_b * N0
    for l in range(3, L):
        # l = 3, 4, ..., L-1
        c_l_1 = c_l    # c_{l-1}
        
        G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, l, M, gamma = 0.8)
        sample_numbers += add_num
        
        # Burn-in
        G = G[L_b * N0:]
        print("len(G): ", len(G)) # Debug
        theta_ls = theta_ls[L_b * N0:][:]
        
        c_l = cp.percentile(G, P)
        print('c', l, ' = ', c_l)
        
        mask = G <= c_l
        G = G[mask][:N0]
        theta_ls = theta_ls[mask][:N0][:]
        
        # if c_l < 0:0 sample_numbers
        
        G_, _, add_num = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, l, M, gamma = 0.8)
        sample_numbers += add_num
        G_ = G_[L_b * N0:]
        
        denominator *= cp.mean(G_ <= c_l_1)
        # print('denominator =', denominator)
    
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, L, M, gamma = 0.8)
    sample_numbers += add_num
    G = G[L_b * N0:]
    
    return p0 ** L * cp.mean(G <= 0) / denominator, sample_numbers

if __name__ == "__main__":
    from tqdm import tqdm

    np.random.seed(26)
    error_list = []
    cost_list = []

    for N in [1000, 4000, 10000]:
        failure_probability, sample_numbers = mle(N, 0)
        
        p_f = failure_probability
        error = rRMSE(p_f)
        error_list.append(error)
        
        exp = 2 ** np.linspace(3, 9, num=7)    # We start with h = 2^(-3)
        cost = np.sum(sample_numbers * exp)
        cost_list.append(cost)

        print(f"Failure probability: {p_f:.2e}")
        print(f"Error: {error:.2e}")
        print(f"Cost: {cost:.2e}\n")
    
    np.save("mle_sr_gpu_error_list.npy", error_list)
    np.save("mle_sr_gpu_cost_list.npy", cost_list)
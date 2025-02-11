from fenics import *
import numpy as np
import cupy as cp
import multiprocessing
from tqdm import tqdm
import os

# Set PETSc options to suppress solver output (and write to file)
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    difference = p_hat - 7.23e-05
    expaction = np.mean(difference ** 2)
    return np.sqrt(expaction) / 7.23e-05


# Precompute eigenvalues and eigenfunctions (on the GPU)
M = 150
x_cpu = np.linspace(0, 1, 1000)
x = cp.asarray(x_cpu) # Transfer to GPU
beta = 1 / 0.01
eigenvalues = cp.array([2 * beta / ((m * np.pi)**2 + beta**2) for m in range(1, M + 1)])
eigenfunctions = cp.array([[np.sqrt(2 * (m * np.pi)**2 / (2 * beta + (m * np.pi)**2 + beta**2)) * cp.cos(m * np.pi * xi) +
                             np.sqrt(2 * beta**2 / (2 * beta + (m * np.pi)**2 + beta**2)) * cp.sin(m * np.pi * xi)
                             for xi in x] for m in range(1, M + 1)])

def kl_expan(theta):
    theta = cp.asarray(theta) # Ensure theta is on the GPU
    mu = cp.array(-0.5 * np.log(1.01))
    sigma = cp.array(np.sqrt(np.log(1.01)))
    log_a_x = mu + sigma * cp.sum(cp.sqrt(eigenvalues)[:, np.newaxis] * eigenfunctions * theta[:, np.newaxis], axis=0)
    return cp.asnumpy(cp.exp(log_a_x)) 


def IoQ(a_x, n_grid):
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)

    a = Function(V)
    a_values = np.interp(mesh.coordinates().flatten(), np.linspace(0, 1, len(a_x)), a_x)
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


def mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma=0.8):
    N0 = len(G)
    for i in range(N - N0):
        seed = thetas[i:i + 1]
        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1, M))
        u_h = IoQ(kl_expan(candidate[0]), n_grid)
        if u_max - u_h <= c_l:
            G = np.append(G, u_max - u_h)
            thetas = np.append(thetas, candidate, axis=0)
        else:
            G = np.append(G, G[i])
            thetas = np.append(thetas, seed, axis=0)
    return G, thetas


def subset_simulation(N, M = 150, p0 = 0.1, u_max = 0.535, n_grid = 256, gamma = 0.8, L = 4):
    N0 = int(N * p0)
    
    l = 1
    G = np.array([])
    sample_number = np.zeros(L)
    
    thetas = np.random.normal(0, 1, (N, M))
    for i in range(N):
        theta = thetas[i]
        u_1 = IoQ(kl_expan(theta), n_grid)
        g = u_max - u_1
        G = np.append(G, g)
        
    sample_number[0] = N
    
    # Compute the threshold value
    c_l = np.percentile(G, 100 * p0)
    # print(f"c_{l}: ", c_l)
    
    if c_l < 0:
        return len(G) / N, sample_number
    
    mask = G <= c_l
    G = G[mask][:N0]
    thetas = thetas[mask][:N0][:]
    p_f = p0
    
    for l in range(2, L):
        G, thetas = mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma)
        sample_number[l-1] = N - N0
        
        c_l = np.percentile(G, 100 * p0)
        # print(f"c_{l}: ", c_l)
        
        if c_l < 0:
            mask = G <= 0
            p_f *= mask.mean()
            return p_f, sample_number
        
        mask = G <= c_l
        G = G[mask][:N0]
        thetas = thetas[mask][:N0][:]
        p_f *= p0
        
    G, _ = mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma)
    sample_number[L-1] = N - N0
    mask = G <= 0
    p_f *= mask.mean()
    
    return p_f, sample_number


if __name__ == "__main__":

    np.random.seed(26)
    error_list = []
    cost_list = []


    for N in tqdm([500, 1000, 1500, 3500, 7000, 10000], desc="Subset Simulations"):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap(subset_simulation, [N] * 100), total=100, desc=f"N = {N}"))

        failure_probabilities = [r[0] for r in results]
        costs = [np.sum(r[1]) for r in results]


        err = rRMSE(failure_probabilities)
        error_list.append(err)
        print("Relative root mean square error: ", err)
        cost_list.append(np.mean(costs))
        print("Average cost: ", np.mean(costs))

    np.save("SubS_error_list.npy", error_list)
    np.save("SubS_cost_list.npy", cost_list)
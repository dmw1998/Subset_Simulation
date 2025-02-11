from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import os

# Set PETSc options to suppress solver output (and write to file)
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    difference = p_hat - 1.6e-04
    expaction = np.mean(difference ** 2)
    return np.sqrt(expaction) / 1.6e-04


# Precompute eigenvalues and eigenfunctions
M = 150  # Example value
x = np.linspace(0, 1, 1000)
beta = 1 / 0.01
eigenvalues = np.array([2 * beta / ((m * np.pi)**2 + beta**2) for m in range(1, M + 1)])
eigenfunctions = np.array([[np.sqrt(2 * (m * np.pi)**2 / (2 * beta + (m * np.pi)**2 + beta**2)) * np.cos(m * np.pi * xi) +
                             np.sqrt(2 * beta**2 / (2 * beta + (m * np.pi)**2 + beta**2)) * np.sin(m * np.pi * xi)
                             for xi in x] for m in range(1, M + 1)])

def kl_expan(theta):
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    log_a_x = mu + sigma * np.sum(np.sqrt(eigenvalues)[:, np.newaxis] * eigenfunctions * theta[:, np.newaxis], axis=0)
    return np.exp(log_a_x)


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



def subset_simulation(N, M = 150, p0 = 0.1, u_max = 0.535, n_grid = 1024, gamma = 0.8, L = 4):
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
    print(f"c_{l}: ", c_l)
    
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
        print(f"c_{l}: ", c_l)
        
        if c_l < 0:
            mask = G <= 0
            p_f *= mask.mean()
            return p_f, sample_number
        
        # shuffling the samples
        Indices = np.arange(len(G))
        np.random.shuffle(Indices)
        
        G = G[Indices]
        thetas = thetas[Indices]
        
        mask = G <= c_l
        G = G[mask][:N0]
        thetas = thetas[mask][:N0][:]
        p_f *= p0
        
    G, _ = mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma)
    sample_number[L-1] = N - N0
    mask = G <= 0
    p_f *= mask.mean()
    
    return p_f, sample_number


def subset_simulation_with_seed(args):
    N, seed_offset = args
    np.random.seed(seed_offset)  # Each worker gets a unique seed
    return subset_simulation(N)


if __name__ == "__main__":
    np.random.seed(66)

    error_list = []
    cost_list = []

    for N in [100]:
        # failure_probabilities = []
        # cost_on_each_level = []
        # cost = 0
            
        args = [(N, np.random.randint(10, 1000)) for _ in range(100)]
        
        with multiprocessing.Pool(processes=12) as pool:
            results = list(tqdm(pool.imap(subset_simulation_with_seed, args), total=100, desc=f"N = {N}"))

        failure_probabilities = [r[0] for r in results]
        # cost_on_each_level = np.mean([r[1] for r in results]) * 1024
        costs = [np.sum(r[1]) * 1024 for r in results]   # 1024 is the cost of one IoQ evaluation since the mesh has 1024 grids and we are working on the unit interval
        
        # for res in results:
        #     failure_probabilities.append(res[0])
        #     cost_on_each_level.append(res[1] * 1024)
        #     cost += np.sum(res[1])
        
        err = rRMSE(failure_probabilities)
        error_list.append(err)
        print("Relative root mean square error: {:.2e}".format(err))
        
        costs = np.mean(costs)
        cost_list.append(costs)
        print("Average cost: {:.2e}\n".format((costs)))
        
        # cost_on_each_level = np.mean(cost_on_each_level, axis=0)
        
        # np.save(f"SubS_cost_on_each_level1024_{N}.npy", cost_on_each_level)
        # np.save("SubS_error_list1024.npy", error_list)
        # np.save("SubS_cost_list1024.npy", cost_list)
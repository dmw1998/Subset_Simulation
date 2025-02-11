# A multileve estimator with fully refinement for the 1D example
from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import os

# Set PETSc options to suppress solver output (and write to file)
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def rRMSE(failure_probability):
    
    difference = np.array(failure_probability) - 1.6e-04
    
    return np.sqrt(np.mean(difference ** 2)) / 1.6e-04


def kl_expan(theta):
    M = len(theta)

    x = np.linspace(0, 1, 1000)

    beta = 1 / 0.01

    def eigenvalue(m):
        w = m * np.pi
        return 2 * beta / (w ** 2 + beta ** 2)

    def eigenfunction(m, x):
        w = m * np.pi
        A = np.sqrt(2 * w ** 2 / (2 * beta + w ** 2 + beta ** 2))
        B = np.sqrt(2 * beta ** 2 / (2 * beta + w ** 2 + beta ** 2))
        return A * np.cos(w * x) + B * np.sin(w * x)

    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))

    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m + 1)) * eigenfunction(m + 1, x) * theta[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)

    return a_x


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
    solve(a_form == L, u_h, bc, solver_parameters={
        'linear_solver': 'cg',
        'preconditioner': 'ilu'}
        )
    return u_h(1)

def mh_sampling(N, G, thetas, c_l, l, u_max, n_grid, M = 150, gamma=0.8):
    N0 = len(G)
    n_grid_0 = n_grid
    samples_numbers = np.zeros(7)
    
    for i in range(N - N0):
        tol = 0.31
        n_grid = n_grid_0
        seed = thetas[i : i+1]
        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1, M))
        g = u_max - IoQ(kl_expan(candidate[0]), n_grid)
        samples_numbers[0] += 1
        
        for j in range(1, l):
            if tol >= np.abs(g - c_l):
                tol *= 0.31
                n_grid *= 2
                g = u_max - IoQ(kl_expan(candidate[0]), n_grid)
                samples_numbers[j] += 1
            else:
                break
            
        if g <= c_l:
            G = np.append(G, g)
            thetas = np.concatenate([thetas, candidate], axis=0)
        else:
            G = np.append(G, G[i])
            thetas = np.concatenate([thetas, seed], axis=0)
            
    return G, thetas, samples_numbers



def mle(args):
    N, seed = args
    np.random.seed(seed)  # Unique seed for each worker
    # N = args
    
    p0 = 0.25
    M = 150
    L_b = 10
    u_max = 0.535
    n_grid = 32
    L = 7
    
    # input:
    # N: number of samples
    # p0: probability of failure
    # M: length of theta
    # L_b: burn-in length
    # u_max: critical value
    # n_grid: number of grid points
    # L: number of levels, L should larger than 3
    
    # output:
    # p_f: final probability of failure
    
    N0 = int(N * p0)
    P = 100 * p0
    
    # Generate initial samples
    sample_numbers = np.zeros(L)
    theta_ls = np.random.normal(0, 1, (N, M))
    G = np.zeros(N)
    
    for i in range(N):
        g = IoQ(kl_expan(theta_ls[i]), n_grid)
        G[i] = u_max - g
    
    sample_numbers[0] += N
    
    # Determine the threshold value c_l
    c_l = np.percentile(G, P)
    # print('c_1 = ', c_l)
    
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return len(G) / N, sample_numbers
    else:
        denominator = 1
    
    # l < 3, set L_b = 0    
    # Sampling without burn-in
    c_l_1 = c_l    # c_{l-1}
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    # print(len(G))
    c_l = np.percentile(G, P)
    # print('c_2 = ', c_l)
    
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    sample_numbers += add_num
    
    G_, _, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    sample_numbers += add_num
    
    denominator *= len(G_ <= c_l_1) / N
    # print('denominator =', denominator)
    
    if c_l <= 0:
        return p0 * np.mean(G <= 0) / N / denominator, sample_numbers
    
    # l > 2, set L_b = L_b
    # Sampling with burn-in
    N += L_b * N0
    for l in range(3, L):
        c_l_1 = c_l    # c_{l-1}
        G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        
        G = G[L_b * N0:]
        # print(len(G))
        theta_ls = theta_ls[L_b * N0:, :]
        
        c_l = np.percentile(G, P)
        # print('c', l, ' = ', c_l)
        
        mask = G <= c_l
        G = G[mask][:N0]
        # print(len(G))
        theta_ls = theta_ls[mask][:N0, :]
        sample_numbers += add_num
        
        if c_l <= 0:
            return p0 ** (l-1) * np.mean(G <= 0) / denominator, sample_numbers
        
        G_, _, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        G_ = G_[L_b * N0:]
        sample_numbers += add_num
        
        denominator *= np.mean(G_ <= c_l_1)
        # print('denominator =', denominator)
    
    G, _, add_num = mh_sampling(N, G, theta_ls, c_l, L, u_max, n_grid)
    G = G[L_b * N0:]
    sample_numbers += add_num
    
    return p0 ** (L-1) * np.mean(G <= 0) / denominator, sample_numbers

if __name__ == "__main__":
    np.random.seed(361)
    error_list = []
    cost_list = []
    sample_number_list = []

    for N in [100, 200, 400, 800, 1600]:
        with multiprocessing.Pool(processes=12) as pool:
            seeds = np.random.randint(100, 1000, 100)
            args = [(N, seed) for seed in seeds]  # Unique seed offset for each worker
            results = list(tqdm(pool.imap(mle, args), total=100, desc=f"N = {N}"))
            
        failure_probabilities = [res[0] for res in results]
        sample_numbers = [res[1] for res in results]
        sample_numbers = np.mean(sample_numbers, axis=0)
        print(sample_numbers)
        sample_number_list.append(sample_numbers)
            
        p_f = np.mean(failure_probabilities)
        error = rRMSE(failure_probabilities)
        error_list.append(error)
        
        exp = 0.31 ** (-2*np.linspace(5, 5+6, num=7))    # We start with h = 2^(-5)
        cost = np.sum(sample_numbers * exp)
        cost_list.append(cost)

        print(f"Failure probability: {p_f:.2e}")
        print(f"Error: {error:.2e}")
        print(f"Cost: {cost:.2e}\n")
    
        np.save("mle_sr_sample_numbers32.npy", sample_number_list)
        np.save("mle_sr_error_list32.npy", error_list)
        np.save("mle_sr_cost_list32.npy", cost_list)
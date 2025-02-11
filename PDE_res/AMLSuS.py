from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import logging
import os

# Configure logging and computing environment
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Global constants
TARGET_RISK = 1.6e-04
RELAXATION_FACTOR = 0.31
U_MAX = 0.535
BETA = 1 / 0.01
MU = -0.5 * np.log(1.01)
SIGMA = np.sqrt(np.log(1.01))

def rRMSE(failure_probability):
    difference = np.array(failure_probability) - TARGET_RISK
    return np.sqrt(np.mean(difference ** 2)) / TARGET_RISK

def delta(p_hat, I):
    if p_hat == 0:
        return 10
    elif p_hat == 1:
        return 0
    
    N = len(I)
    # if N < 101:
    #     return 10 
    
    max_lag = N//2
    autocorr = np.correlate(I - np.mean(I), I - np.mean(I), mode='full')[N-1-max_lag:N+max_lag]
    autocorr = autocorr / (np.var(I) * N)
    
    ess = N / (1 + 2 * np.sum(autocorr[1:max_lag+1]))
    return np.sqrt(p_hat * (1 - p_hat) / ess)
    
def compute_y(y_L, L, gamma):
    y = np.zeros(L)
    
    for l in range(L-2, -1, -1):
        y[l] = y[l+1] + gamma**(l+3) + gamma**(l+4)
        
    return y
    
def kl_expan(theta):
    M = len(theta)
    x = np.linspace(0, 1, 1000)
    
    m_vals = np.arange(1, M+1).reshape(-1, 1)  # shape: (M,1)
    w = m_vals * np.pi  # shape: (M,1)
    
    lambda_vals = 2 * BETA / (w**2 + BETA**2)  # shape: (M,1)
    sqrt_lambda = np.sqrt(lambda_vals)
    
    A = np.sqrt(2 * w**2 / (2 * BETA + w**2 + BETA**2))
    B = np.sqrt(2 * BETA**2 / (2 * BETA + w**2 + BETA**2))
    phi = A * np.cos(w * x) + B * np.sin(w * x)  # shape: (M, len(x))
    
    log_a_x = MU + SIGMA * np.sum(sqrt_lambda * phi * theta.reshape(-1, 1), axis=0)
    a_x = np.exp(log_a_x)
    
    return a_x


def IoQ(a_x, n_grid):
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)

    coordinates = V.tabulate_dof_coordinates().reshape(-1)
    a_values = np.interp(coordinates, np.linspace(0, 1, len(a_x)), a_x)
    a = Function(V)
    a.vector()[:] = a_values
    
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    bc = DirichletBC(V, Constant(0.0), boundary)

    u = TrialFunction(V)
    v = TestFunction(V)

    a_form = inner(a * grad(u), grad(v)) * dx
    L = Constant(1.0) * v * dx

    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L, u_h, bc, solver_parameters={
        'linear_solver': 'lu',
        # 'preconditioner': 'ilu'
    })
    
    return u_h(1.0)


def AMLSuS(args, M = 150, u_max = 0.535, n_grid = 128, gamma = 0.8, L = 5):
    tol, seed = args
    np.random.seed(seed)
    
    y = compute_y(0, L, gamma=0.31)
    # print(y)
    
    G = np.array([])
    # thetas = np.array([])
    sample_numbers = np.zeros(L)
    err = 10
    
    N = 99
    thetas = np.random.normal(0, 1, (N,M))
    for i in range(N):
        theta = thetas[i:i+1]
        u_1 = IoQ(kl_expan(theta[0]), n_grid)
        g = u_max - u_1
        G = np.append(G, g)
        
    sample_numbers[0] += N
    
    mask = G <= y[0]
    p_hat = np.mean(mask)
    
    if p_hat == 0 or p_hat == 1:
        err = 10
    else:
        err = p_hat * (1 - p_hat) / sample_numbers[0]
    
    # Level 1
    # tol /= np.sqrt(L)
    while err > tol:
        theta = np.random.normal(0, 1, (1,M))
        G_new = u_max - IoQ(kl_expan(theta[0][:]), n_grid)
        sample_numbers[0] += 1
        
        G = np.append(G, G_new)
        thetas = np.append(thetas, theta, axis=0)
        
        mask = G <= y[0]
        p_hat = np.mean(mask)
        
        if p_hat == 0:
            err = 10
        else:
            err = p_hat * (1 - p_hat) / sample_numbers[0]
            
        # if sample_numbers[0] % 1000 == 0:
        #     print(f"Level 1: iteration = {sample_numbers[0]:.0f}, err = {err:.2e}, p_hat = {p_hat:.2e}")
            # if p_hat == 1:
            #     err = 0
            
    p_f = p_hat
    # print(f"Level 1: p_f = {p_f:.2e}, err = {err:.2e}, sample number = {sample_numbers[0]:.0f}")
    
    # Level 2 to L
    # tol /= np.sqrt(L)
    for l in range(1, L):
        n_grid *= 2
        G = G[mask][:1]
        thetas = thetas[mask][:1, :]
        err = 10
        
        while err > tol:
            # print(thetas[-1])
            theta = gamma * thetas[-1] + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1,M))
            G_new = u_max - IoQ(kl_expan(theta[0][:]), n_grid)
            
            if G_new <= y[l]:
                G = np.append(G, G_new)
                thetas = np.append(thetas, theta, axis=0)
            else:
                G = np.append(G, G[-1])
                thetas = np.append(thetas, thetas[-1:], axis=0)
                
            sample_numbers[l] += 1
            
            mask = G <= y[l]
            p_hat = np.mean(mask)
            
            err = delta(p_hat,mask)
            if sample_numbers[l] % 1000 == 0:
                print(f"Level {l+1}: iteration = {sample_numbers[l]:.0f}, err = {err:.2e}, p_hat = {p_hat:.2e}")
                # if p_hat == 1:
                #     err = 0
            
        p_f *= p_hat
        # print(f"Level {l+1}: p_f = {p_f:.2e}, p_hat = {p_hat:.2e}, err = {err:.2e}, sample number = {sample_numbers[l]:.0f}")
        
    return p_f, sample_numbers

if __name__ == "__main__":
    np.random.seed(123)
    
    error_list = []
    cost_list = []
    
    for tol in [8e-2, 6e-2, 4e-2]:
        seeds = np.random.randint(100, 10000, 100)
        args = [(tol, seed) for seed in seeds]
        with multiprocessing.Pool(processes=12) as pool:
            results = list(tqdm(pool.imap(AMLSuS, args), total=100, desc=f"tol = {tol:.2e}"))
            
        failure_probabilities = [res[0] for res in results]
        sample_numbers_arr = [res[1] for res in results]
        sample_numbers_mean = np.mean(sample_numbers_arr, axis=0)
            
        error = rRMSE(failure_probabilities)
        error_list.append(error)
        
        exp = RELAXATION_FACTOR ** (-2 * np.linspace(7, 7+4, num=5))
        cost = np.sum(sample_numbers_mean * exp)
        cost_list.append(cost)
            
        print(f"Average error: {error:.2e}")
        print(f"Average cost: {cost:.2e}")
        print(f"Average sample numbers: {sample_numbers_mean}")
    
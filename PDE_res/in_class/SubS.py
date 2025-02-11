from fenics import *
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
set_log_level(50)  # 禁用FEniCS日志

TARGET_RISK = 1.6e-4
BETA = 100.0  # 1/0.01
MU = -0.5 * np.log(1.01)
SIGMA = np.sqrt(np.log(1.01))

class SubsetSimulation:
    def __init__(self, params):
        self.params = params
        self.initialized = False

    def initialize(self):
        if self.initialized: 
            return

        self.N = self.params['N']
        self.M = self.params['M']
        self.p0 = self.params['p0']
        self.L = self.params['L']
        self.u_max = self.params['u_max']
        self.n_grid = self.params['n_grid']
        self.gamma = self.params['gamma']
        
        # 初始化FEniCS
        self.mesh = UnitIntervalMesh(self.n_grid)
        self.V = FunctionSpace(self.mesh, "P", 1)
        self.x_coords = self.mesh.coordinates().flatten()
        self.a = Function(self.V)
        self.bc = DirichletBC(self.V, Constant(0.0), "x[0] < DOLFIN_EPS")
        
        # 预组装矩阵
        u, v = TrialFunction(self.V), TestFunction(self.V)
        self.a_form = inner(self.a * grad(u), grad(v)) * dx
        self.L_form = Constant(1.0) * v * dx
        self.u_h = Function(self.V)
        
        self.initialized = True

    def kl_expan(self, theta):
        x = self.x_coords
        m = np.arange(1, self.M+1).reshape(-1, 1)
        w = m * np.pi
        lambda_ = 2 * BETA / (w**2 + BETA**2)
        
        # 计算特征函数
        A = np.sqrt(2 * w**2 / (w**2 + BETA**2 + 2*BETA))
        B = np.sqrt(2 * BETA**2 / (w**2 + BETA**2 + 2*BETA))
        phi = A*np.cos(np.outer(w, x)) + B*np.sin(np.outer(w, x))
        
        # 组合KL展开
        log_a = MU + SIGMA * np.sum(np.sqrt(lambda_) * phi * theta.reshape(-1, 1), axis=0)
        return np.exp(log_a)

    def solve_pde(self, a_values):
        self.a.vector()[:] = a_values
        solve(self.a_form == self.L_form, self.u_h, self.bc)
        return self.u_h(1)

    def run(self):
        self.initialize()
        np.random.seed(self.params['seed'])
        
        N0 = int(self.N*self.p0)
        
        # initialization l = 1
        thetas = np.random.normal(size=(self.N, self.M))
        G = np.array([self.u_max - self.solve_pde(self.kl_expan(theta)) 
                     for theta in thetas])
        
        p_f = 1.0
        samples = [self.N]
        
        # iteration l = 2, 3, ..., L
        for level in range(1, self.L):
            # 确定阈值
            c_l = np.percentile(G, 100*self.p0)
            if c_l <= 0:
                p_f *= np.mean(G <= 0)
                samples.append(0)
                break
            
            # 选择种子样本
            mask = G <= c_l
            G = G[mask][:N0]
            thetas = thetas[mask][:N0]
            p_f *= self.p0
            
            # MH采样
            G, thetas = self.mh_sampling(G, thetas, c_l)
            samples.append(self.N - N0)
        
        return p_f, sum(samples)

    def mh_sampling(self, G, thetas, c_l):
        N0 = len(G)
        
        for i in range(self.N - N0):
            seed = thetas[i:i+1]
            candidate = self.gamma*seed + np.sqrt(1-self.gamma**2)*np.random.normal(size=self.M)
            u = self.solve_pde(self.kl_expan(candidate))
            g = self.u_max - u
            
            if g <= c_l:
                G = np.append(G, g)
                thetas = np.append(thetas, candidate, axis=0)
            else:
                G = np.append(G, G[i])
                thetas = np.append(thetas, seed, axis=0)
        
        return G, thetas

def worker(params):
    sim = SubsetSimulation(params)
    return sim.run()

if __name__ == "__main__":
    base_params = {
        'M': 150,
        'p0': 0.1,
        'u_max': 0.535,
        'n_grid': 1024,
        'gamma': 0.8,
        'L': 4,
        'seed': None
    }
    
    error_list = []
    cost_list = []
    for N in [100, 200, 400, 800]:
        base_params['seed'] = 42  # 各进程内部使用独立种子
        params_list = [dict(base_params, N=N, seed=s) 
                      for s in np.random.randint(10000, size=100)]
        
        with mp.Pool(12) as pool:
            p_fs, costs = zip(*list(tqdm(
                pool.imap(worker, params_list),
                total=100, desc=f"N={N}"
            )))
        
        rrms = np.sqrt(np.mean(np.array(p_fs)-TARGET_RISK)**2)/TARGET_RISK  # relative root mean square error rRMSE
        avg_cost = np.mean(costs) * 0.31**(-20) # set q = 2 with l = 10
        print(f"N={N}: rRMSE={rrms:.2e}, Cost={avg_cost:.2e}\n")
        
        error_list.append(rrms)
        cost_list.append(avg_cost)
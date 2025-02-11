from fenics import *
import numpy as np
import logging
import os

# -------------------------
# 环境配置
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# -------------------------
# 全局常量 (根据问题定义)
# -------------------------
TARGET_RISK = 1.6e-04
U_MAX = 0.535
BETA = 1 / 0.01
MU_LN_A = -0.5 * np.log(1.01)
SIGMA_LN_A = np.sqrt(np.log(1.01))
MIN_SAMPLES = 500  # 每层最小样本数
EPSILON = 1e-8     # 正则化参数

# -------------------------
# 核心函数 (理论指导实现)
# -------------------------

def kl_expansion(theta, x_points=1000):
    """生成随机场数组"""
    M = len(theta)
    x = np.linspace(0, 1, x_points)
    
    m = np.arange(1, M+1)[:, None]
    w = m * np.pi
    lambda_m = 2 * BETA / (w**2 + BETA**2)
    sqrt_lambda = np.sqrt(lambda_m)
    
    A = np.sqrt(2 * w**2 / (2 * BETA + w**2 + BETA**2))
    B_term = np.sqrt(2 * BETA**2 / (2 * BETA + w**2 + BETA**2))
    phi = A * np.cos(w * x) + B_term * np.sin(w * x)
    
    log_a = MU_LN_A + SIGMA_LN_A * np.sum(sqrt_lambda * phi * theta[:, None], axis=0)
    return np.exp(log_a)

def solve_pde(a_x, n_grid):
    """PDE求解函数（仅接受随机场数组输入）"""
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    # 插值生成随机场
    a_func = Function(V)
    coordinates = V.tabulate_dof_coordinates().flatten()
    a_values = np.interp(coordinates, np.linspace(0, 1, len(a_x)), a_x)
    a_func.vector()[:] = a_values
    
    # 边界条件
    bc = DirichletBC(V, Constant(0.0), "near(x[0], 0)")
    
    # 变分问题
    u = TrialFunction(V)
    v = TestFunction(V)
    a_form = inner(a_func * grad(u), grad(v)) * dx
    L_form = Constant(1.0) * v * dx
    
    # 求解
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L_form, u_h, bc, solver_parameters={
        'linear_solver': 'cg',
        'preconditioner': 'hypre_amg'
    })
    return u_h(1.0)

def verify_accuracy(a_x, n_grid):
    """精度验证函数（基于随机场数组）"""
    # 计算两种网格下的解
    u_coarse = solve_pde(a_x, n_grid)
    u_fine = solve_pde(a_x, 2*n_grid)
    return abs(u_coarse - u_fine) < 1e-6

def compute_thresholds(L, gamma):
    """严格遵循引理3.1的阈值生成"""
    c = np.zeros(L+1)  # c[0] unused
    c[L] = 0.0  # 最后一层阈值
    
    for l in range(L-1, 0, -1):
        c[l] = c[l+1] + gamma**l + gamma**(l+1)
    
    # 调整首层阈值确保包含性
    c[1] = max(c[1], U_MAX * 1.1)
    return c[1:L+1]

def robust_p_hat(I):
    """正则化条件概率估计"""
    n = len(I)
    k = np.sum(I) + EPSILON
    return k / (n + 2*EPSILON), (k*(n-k)) / ((n + 2*EPSILON)**3)  # (估计值, 方差)

def autocorr_factor(I, p_hat):
    n = len(I)
    if n < 100 or p_hat in (0, 1):
        return 1.0
    max_lag = min(1000, n//2)
    I_centered = I - p_hat
    autocorr = np.correlate(I_centered, I_centered, mode='full')[n-1-max_lag:n+max_lag]
    cutoff = np.where(np.abs(autocorr) < 0.05)[0][0] if np.any(np.abs(autocorr) < 0.05) else max_lag
    return 1 + 2 * np.sum(autocorr[1:cutoff+1])

class AdaptiveMLSuS:
    def __init__(self, L=4, gamma=0.3, base_grid=64, tol=0.1):
        self.L = L
        self.gamma = gamma
        self.base_grid = base_grid
        self.tol = tol
        self.thresholds = compute_thresholds(L, gamma)
        self.p_levels = []
        self.var_levels = []
        self.cost = 0
        
    def run(self):
        for level in range(self.L):
            current_grid = self.base_grid * (2 ** level)
            self._process_level(level+1, current_grid)
        return self._final_result()
    
    def _process_level(self, level, n_grid):
        logging.info(f"\n=== Level {level} ===")
        
        if level == 1:
            samples = self._level1_sampling(n_grid)
        else:
            samples = self._mcmc_sampling(level, n_grid)
            
        # 概率估计
        I = np.array([g <= self.thresholds[level-1] for _, g in samples])
        p_hat, var = robust_p_hat(I)
        self.p_levels.append(p_hat)
        self.var_levels.append(var * autocorr_factor(I, p_hat))
        self.cost += len(samples) * n_grid
        
        logging.info(f"Conditional p_{level} = {p_hat:.3e} ± {np.sqrt(var):.1e}")
        
    def _level1_sampling(self, n_grid):
        samples = []
        while len(samples) < MIN_SAMPLES:
            theta = np.random.randn(150)
            a_x = kl_expansion(theta)
            if verify_accuracy(a_x, n_grid):  # 先验证精度
                G = U_MAX - solve_pde(a_x, n_grid)
                samples.append((theta, G))
        return samples
    
    def _mcmc_sampling(self, level, n_grid):
        """后续层：MCMC采样"""
        # 从前一层获取种子
        prev_samples = [s for s in self.samples if s[1] <= self.thresholds[level-2]]
        theta = prev_samples[np.argmin([s[1] for s in prev_samples])][0]
        
        chain = []
        for _ in range(MIN_SAMPLES):
            theta, accepted = self._mcmc_step(theta, level, n_grid)
            chain.append((theta, U_MAX - solve_pde(kl_expansion(theta), n_grid)))
            
        return chain
    
    def _mcmc_step(self, theta_prev, level, n_grid):
        """MCMC单步"""
        # 生成候选样本
        proposal = self.gamma*theta_prev + np.sqrt(1-self.gamma**2)*np.random.randn(150)
        
        # 确保满足前层条件
        G_proposal = U_MAX - solve_pde(kl_expansion(proposal), n_grid)
        if G_proposal > self.thresholds[level-2]:
            return theta_prev, False
        
        # 计算接受概率
        G_prev = U_MAX - solve_pde(kl_expansion(theta_prev), n_grid)
        alpha = min(1, (G_proposal <= self.thresholds[level-1]) / 
                    (G_prev <= self.thresholds[level-1]))
        
        # 接受/拒绝
        if np.random.rand() < alpha:
            return proposal, True
        return theta_prev, False
    
    def _final_result(self):
        """计算最终结果"""
        p_failure = np.prod(self.p_levels)
        total_var = np.sum(np.array(self.var_levels) / np.array(self.p_levels)**2)
        total_error = np.sqrt(total_var)
        
        logging.info(f"\n=== Final Result ===")
        logging.info(f"Estimated P_f: ({p_failure:.3e} ± {total_error:.1e})")
        logging.info(f"Total cost: {self.cost} grid points")
        return p_failure, total_error

# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    amlsus = AdaptiveMLSuS(L=4, gamma=0.3)
    p_f, error = amlsus.run()
    logging.info(f"Final result: {p_f:.3e} ± {error:.1e}")
    logging.info(f"Relative error: {abs(p_f-TARGET_RISK)/TARGET_RISK*100:.2f}%")
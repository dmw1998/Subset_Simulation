from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import os
import logging

# -------------------------
# 配置日志、环境变量及常量
# -------------------------

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 设置 PETSc 选项，关闭求解器输出，并写入文件
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 全局常量
TARGET_RISK = 1.6e-04      # 用于计算相对均方根误差
RELAXATION_FACTOR = 0.31   # 用于控制细化条件
U_MAX = 0.535              # 关键值 u_max
BETA = 1 / 0.01            # KL 展开中用到的 beta
MU = -0.5 * np.log(1.01)     # log-normal 均值参数
SIGMA = np.sqrt(np.log(1.01))  # log-normal 标准差参数

# -------------------------
# 辅助函数
# -------------------------
def rRMSE(failure_probability):
    """
    计算 failure_probability 与 TARGET_RISK 之间的相对均方根误差(rRMSE)
    """
    difference = np.array(failure_probability) - TARGET_RISK
    return np.sqrt(np.mean(difference ** 2)) / TARGET_RISK

def kl_expan(theta):
    """
    向量化的 Karhunen-Loève 展开，用于生成随机场 a(x)
    
    参数:
      theta: 一维数组，长度为 M
      
    返回:
      a_x: 在 [0,1] 上 1000 个点处的随机场数值
    """
    M = len(theta)
    x = np.linspace(0, 1, 1000)
    
    # 构造模式索引，并计算对应角频率 w
    m_vals = np.arange(1, M+1).reshape(-1, 1)  # shape: (M,1)
    w = m_vals * np.pi  # shape: (M,1)
    
    # 计算特征值 lambda 和其平方根
    lambda_vals = 2 * BETA / (w**2 + BETA**2)  # shape: (M,1)
    sqrt_lambda = np.sqrt(lambda_vals)
    
    # 计算特征函数
    A = np.sqrt(2 * w**2 / (2 * BETA + w**2 + BETA**2))
    B = np.sqrt(2 * BETA**2 / (2 * BETA + w**2 + BETA**2))
    phi = A * np.cos(w * x) + B * np.sin(w * x)  # shape: (M, len(x))
    
    # 计算 log_a(x) 并取指数得到 a(x)
    log_a_x = MU + SIGMA * np.sum(sqrt_lambda * phi * theta.reshape(-1, 1), axis=0)
    a_x = np.exp(log_a_x)
    return a_x

def IoQ(a_x, n_grid):
    """
    求解 PDE 并返回解在 x=1 处的值
    
    参数:
      a_x: 随机场 a(x) 的数值数组
      n_grid: 网格划分数，构造 UnitIntervalMesh(n_grid)
      
    返回:
      u_h(1): PDE 求解的数值解在 x=1 处的值
    """
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    # 通过 V.tabulate_dof_coordinates() 获取节点坐标
    coordinates = V.tabulate_dof_coordinates().reshape(-1)
    a_values = np.interp(coordinates, np.linspace(0, 1, len(a_x)), a_x)
    a = Function(V)
    a.vector()[:] = a_values
    
    # 定义边界条件：在 x=0 处
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    bc = DirichletBC(V, u0, boundary)
    
    # 定义弱形式
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    a_form = inner(a * grad(u), grad(v)) * dx
    L_form = f * v * dx
    
    # 求解 PDE
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L_form, u_h, bc, solver_parameters={
        'linear_solver': 'cg',
        # 'preconditioner': 'ilu'
    })
    return u_h(1)

def mh_sampling(N, G, thetas, c_l, l, u_max, n_grid, M=150, gamma=0.8):
    """
    基于 Metropolis-Hastings 方法的选择性细化采样

    参数:
      N: 总样本数
      G: 当前 failure 指标数组
      thetas: 当前 theta 样本数组，形状为 (n_samples, M)
      c_l: 当前层的阈值
      l: 当前层数(整数)
      u_max: 关键值
      n_grid: 初始网格数
      M: theta 的维数(默认150)
      gamma: 用于构造候选样本的比例因子
      
    返回:
      更新后的 G, thetas, 以及各层采样数统计(samples_numbers 数组)
    """
    N0 = len(G)
    n_grid_0 = n_grid
    # 使用列表累积结果，避免在循环中频繁使用 np.append 和 np.concatenate
    G_list = list(G)
    theta_list = list(thetas)
    samples_numbers = np.zeros(7, dtype=int)
    
    for i in range(N - N0):
        tol = RELAXATION_FACTOR
        current_n_grid = n_grid_0
        # 当前样本作为种子
        seed = theta_list[i]  # shape (M,)
        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (M,))
        
        # 计算指标 g = u_max - IoQ(kl_expan(candidate), n_grid)
        g = u_max - IoQ(kl_expan(candidate), current_n_grid)
        samples_numbers[0] += 1
        
        # 细化迭代，根据 g 与阈值 c_l 的距离进行网格细化
        for j in range(1, l):
            if tol >= np.abs(g - c_l):
                tol *= RELAXATION_FACTOR
                current_n_grid *= 2
                g = u_max - IoQ(kl_expan(candidate), current_n_grid)
                samples_numbers[j] += 1
            else:
                break
        
        if g <= c_l:
            G_list.append(g)
            theta_list.append(candidate)
        else:
            # 若候选样本未满足条件，则保留原样本
            G_list.append(G_list[i])
            theta_list.append(seed)
    
    return np.array(G_list), np.array(theta_list), samples_numbers

def mle(args):
    """
    多层估计器 mle_sr 结合选择性细化采样进行失效概率估计

    参数:
      args: 元组 (N, seed)
          N: 样本总数
          seed: 每个进程使用的随机种子
          
    返回:
      (估计的失效概率, 各层采样数统计数组)
    """
    N, seed = args
    np.random.seed(seed)
    
    # 参数设置
    p0 = 0.25
    M = 150
    L_b = 10
    u_max = U_MAX
    n_grid = 128
    L = 7
    
    N0 = int(N * p0)
    P = 100 * p0  # 百分位数
    
    # 生成初始 theta 样本
    theta_ls = np.random.normal(0, 1, (N, M))
    G = np.zeros(N)
    
    sample_numbers = np.zeros(L, dtype=int)
    for i in range(N):
        g = u_max - IoQ(kl_expan(theta_ls[i]), n_grid)
        G[i] = g
    sample_numbers[0] += N
    
    # 计算第一层阈值 c_l(百分位数)
    c_l = np.percentile(G, P)
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    
    # 若阈值为负，则直接返回
    if c_l < 0:
        return len(G) / N, sample_numbers
    else:
        denominator = 1.0
    
    # 第二层采样(无 burn-in)
    c_l_prev = c_l
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    c_l = np.percentile(G, P)
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    sample_numbers += add_num

    G_temp, _, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    sample_numbers += add_num
    denominator *= np.mean(G_temp <= c_l_prev)
    
    if c_l <= 0:
        return p0 * np.mean(G <= 0) / N / denominator, sample_numbers

    # 第三层及之后的采样(带 burn-in)
    N_extended = N + L_b * N0
    N = N_extended
    for l in range(3, L):
        c_l_prev = c_l
        G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        # 丢弃 burn-in 样本
        G = G[L_b * N0:]
        theta_ls = theta_ls[L_b * N0:, :]
        c_l = np.percentile(G, P)
        mask = G <= c_l
        G = G[mask][:N0]
        theta_ls = theta_ls[mask][:N0, :]
        sample_numbers += add_num

        if c_l <= 0:
            return p0 ** (l-1) * np.mean(G <= 0) / denominator, sample_numbers

        G_temp, _, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        G_temp = G_temp[L_b * N0:]
        sample_numbers += add_num
        denominator *= np.mean(G_temp <= c_l_prev)
    
    # 最后一层采样
    G, _, add_num = mh_sampling(N, G, theta_ls, c_l, L, u_max, n_grid)
    G = G[L_b * N0:]
    sample_numbers += add_num
    
    return p0 ** (L-1) * np.mean(G <= 0) / denominator, sample_numbers

# -------------------------
# 主程序入口
# -------------------------
if __name__ == "__main__":
    np.random.seed(68)
    error_list = []
    cost_list = []
    sample_number_list = []

    # 对于不同的样本规模 N 进行实验
    for N in [200, 400, 800]:
    # for N in [1000]:
        with multiprocessing.Pool(processes=12) as pool:
            seeds = np.random.randint(100, 10000, 100)
            args = [(N, int(seed)) for seed in seeds]  # 为每个子进程指定不同的种子
            results = list(tqdm(pool.imap(mle, args), total=100, desc=f"N = {N}"))
        
        # 从子进程中收集结果
        failure_probabilities = [res[0] for res in results]
        sample_numbers_arr = [res[1] for res in results]
        sample_numbers_mean = np.mean(sample_numbers_arr, axis=0)
        logging.info(f"Sample numbers for N={N}: {sample_numbers_mean}")
        sample_number_list.append(sample_numbers_mean)
        
        p_f = np.mean(failure_probabilities)
        error = rRMSE(failure_probabilities)
        error_list.append(error)
        
        # 估计成本(使用细化比例计算 cost 权重)
        exp = RELAXATION_FACTOR ** (-2 * np.linspace(7, 7+6, num=7)) # set q = 2 with l = 5, ..., 11
        cost = np.sum(sample_numbers_mean * exp)
        cost_list.append(cost)
        
        print(f"Failure probability: {p_f:.2e}")
        print(f"Error: {error:.2e}")
        print(f"Cost: {cost:.2e}\n")
        
        # 保存结果(每次更新)
        np.save("mle_sr_sample_numbers128.npy", sample_number_list)
        np.save("mle_sr_error_list128.npy", error_list)
        np.save("mle_sr_cost_list128.npy", cost_list)

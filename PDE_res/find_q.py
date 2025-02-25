from fenics import *
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
import time
import cProfile
import pstats

BETA = 1 / 0.01            # KL 展开中用到的 beta
MU = -0.5 * np.log(1.01)     # log-normal 均值参数
SIGMA = np.sqrt(np.log(1.01))  # log-normal 标准差参数

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
    solve(a_form == L_form, u_h, bc)
    
    return u_h(1)

def run_profile():
    levels = np.arange(1, 7)  # Test levels l = 6 to 11
    execution_times = []
    
    for l in levels:
        n_grid = 2**(5+l)  # Define grid refinement
        start_time = time.time()
        for _ in range(100):
            theta = np.random.randn(150)
            a_x = kl_expan(theta)
            IoQ(a_x, n_grid)  # Run the PDE solver
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 100
        execution_times.append(avg_time)
        print(f"Level {l}, Grid {n_grid}, Avg Time {avg_time:.4f} sec")

    return levels, execution_times

if __name__ == "__main__":
    np.random.seed(0)
    
    # Run the computation and profile it
    profiler = cProfile.Profile()
    profiler.enable()
    levels, times = run_profile()
    profiler.disable()

    # Print profiling results sorted by cumulative time.
    ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    ps.print_stats(20)
    
    # Optionally, save profiling data to a file:
    ps.dump_stats("profile_stats.prof")
    
    # # Fit a power-law model to the data
    # log_times = np.log(times)
    # # Fit a linear model to log_times versus levels
    # slope, intercept = np.polyfit(levels, log_times / np.log(0.31), 1)
    # q_estimated = -slope  # because our model gives a slope of -q
    # print(f"Estimated q: {q_estimated:.4f}")

    # plt.figure(figsize=(12, 8))
    # plt.scatter(levels, times, marker='o', label="Measured Times", color="tab:blue")
    # plt.plot(levels, np.exp(intercept * np.log(0.31)) * 0.31 ** (-levels * q_estimated), "--", label=f"Estimated: $0.31^{{-{q_estimated:.2f}l}}$", color="tab:red")
    # plt.xlabel("Level l")
    # plt.ylabel("Avg Execution Time (s)")
    # plt.yscale("log")
    # plt.grid()
    # plt.legend()
    # plt.savefig("timing_scaling.png")
    # plt.show()
    
    from scipy.optimize import curve_fit

    # 定义新的拟合模型
    def model(l, A, B, q):
        return A + B * 0.31**(-q * l)

    # 使用非线性拟合
    params, _ = curve_fit(model, levels, times, p0=[1e-2, 1e-2, 0.5])
    A_fit, B_fit, q_fit = params

    print(f"Estimated A: {A_fit:.2e}, Estimated B: {B_fit:.2e}, Estimated q: {q_fit:.4f}")

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(levels, times, marker='o', label="Measured Times", color="tab:blue")
    plt.plot(levels, model(np.array(levels), A_fit, B_fit, q_fit), 
            "--", label=f"Estimated: {A_fit:.2e} + {B_fit:.2e} × $0.31^{{-{q_fit:.2f}l}}$", color="tab:red")
    plt.xlabel("Level l")
    plt.ylabel("Avg Execution Time (s)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.savefig("timing_scaling_fixed.png")
    plt.show()
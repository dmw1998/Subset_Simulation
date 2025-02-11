from MLE_sr import *

np.random.seed(7)
N = 1000

def mle_sr_num(args):
    """修改后的 mle 函数，按层合并多次调用的 add_num"""
    N, seed = args
    np.random.seed(seed)
    
    # 参数设置
    p0 = 0.25
    M = 150
    L_b = 10
    u_max = 0.535
    n_grid = 128
    L = 7
    
    # 初始化层累积数组（形状：7层）
    layer_additions = np.zeros((7, 7), dtype=int)
    
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
    layer_additions[0, 0] = N
    
    # 计算第一层阈值 c_l(百分位数)
    c_l = np.percentile(G, P)
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    denominator = 1.0
    
    # 第二层采样(无 burn-in)
    c_l_prev = c_l
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    layer_additions[1, :] += add_num  # l=2 对应索引1（层数从0开始）
    c_l = np.percentile(G, P)
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    sample_numbers += add_num

    G_temp, _, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    layer_additions[1, :] += add_num  # 同一层级 l=2，累加
    sample_numbers += add_num
    denominator *= np.mean(G_temp <= c_l_prev)
    
    if c_l <= 0:
        return (
            p0 * np.mean(G <= 0) / N / denominator,
            layer_additions  # 返回 7×7 矩阵
        )

    # 第三层及之后的采样(带 burn-in)
    N_extended = N + L_b * N0
    N = N_extended
    for l in range(3, L):
        c_l_prev = c_l
        G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        layer_additions[l-1, :] += add_num  # l=3 对应索引2，依此类推
        G = G[L_b * N0:]
        theta_ls = theta_ls[L_b * N0:, :]
        c_l = np.percentile(G, P)
        mask = G <= c_l
        G = G[mask][:N0]
        theta_ls = theta_ls[mask][:N0, :]
        sample_numbers += add_num

        if c_l <= 0:
            return (
                p0 ** (l-1) * np.mean(G <= 0) / denominator,
                layer_additions
            )

        G_temp, _, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        layer_additions[l-1, :] += add_num  # 同一层级 l，累加
        G_temp = G_temp[L_b * N0:]
        sample_numbers += add_num
        denominator *= np.mean(G_temp <= c_l_prev)
    
    # 最后一层采样
    G, _, add_num = mh_sampling(N, G, theta_ls, c_l, L, u_max, n_grid)
    layer_additions[L-1, :] += add_num  # l=7 对应索引6
    G = G[L_b * N0:]
    sample_numbers += add_num
    
    return (
        p0 ** (L-1) * np.mean(G <= 0) / denominator,
        layer_additions
    )
    
if __name__ == "__main__":
    np.random.seed(42)
    num_runs = 100
    N = 1000

    # 初始化存储所有运行的 7×7 矩阵（形状：100次运行 × 7×7）
    all_layer_matrices = np.zeros((num_runs, 7, 7), dtype=int)
    failure_probabilities = np.zeros(num_runs)

    # 并行运行 100 次
    with multiprocessing.Pool(processes=12) as pool:
        seeds = np.random.randint(100, 10000, num_runs)
        args = [(N, int(seed)) for seed in seeds]
        results = list(tqdm(pool.imap(mle_sr_num, args), total=num_runs, desc="Total Runs"))

    # 提取所有 7×7 矩阵
    for run_idx in range(num_runs):
        failure_probabilities[run_idx] = results[run_idx][0]
        all_layer_matrices[run_idx, :, :] = results[run_idx][1]

    # 保存每个层级的 7 维向量到独立文件
    for l in range(7):
        # 提取所有运行中层级 l 的数据（形状：100次运行 × 7）
        layer_data = all_layer_matrices[:, l, :]
        np.save(f"layer_{l}_additions.npy", layer_data)
        # 计算该层级的平均向量
        layer_mean = np.mean(layer_data, axis=0)
        print(f"Layer_{l}_ave_sample_num: {layer_mean}")

    # 保存完整数据
    np.save("all_layers_additions.npy", all_layer_matrices)
    
    # 计算平均失效概率
    p_f = np.mean(failure_probabilities)
    print(f"Average failure probability: {p_f:.2e}")
    error = rRMSE(failure_probabilities)
    print(f"rRMSE: {error:.2e}")
    
    # 计算平均采样次数
    sample_numbers = np.sum(all_layer_matrices, axis=0)
    print(f"Average sample numbers: {sample_numbers}")
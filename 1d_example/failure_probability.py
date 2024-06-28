# To compute the failure level c_l for a given failure probability p0.

def compute_cl(G, thetas_ls, N, p0, l, L):
    # input:
    # G: approximated IoQ
    # thetas_ls: list of thetas
    # N: number of samples
    # p0: failure probability
    # l: current level
    # L: finest level
    
    # output:
    # G, thetas_ls: updated G and thetas_ls
    # c_l: failure level
    
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_theta_ls = [thetas_ls[i] for i in sorted_indices]
    
    N0 = int(p0 * N)
    c_l = sorted_G[N0-1]
    if c_l < 0 or l == L:
        # When we reach the finest level
        G = [g for g in G if g < 0]
    else:
        G = sorted_G[:N0]
        thetas_ls = sorted_theta_ls[:N0]
    
    return G, thetas_ls, c_l
import numpy as np
import time

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    return np.sqrt(np.mean((p_hat - 7.23e-05)**2)) / 7.23e-05


def delta(p_hat, I):
    if p_hat == 0 or p_hat == 1:
        return 10

    N = len(I)

    gamma_0 = (1 - p_hat) * p_hat

    for m in range(1, N):
        s = (np.mean(I[:-m] * I[m:]) - p_hat**2) * (1 - m/N)

    phi = 2 + 2 * np.sum(s) / gamma_0

    if phi < 0:
        return 10

    # print(f"phi={phi:.3f}")
    return np.sqrt((1 - p_hat) / p_hat / N * (1 + phi))


def adaptive_multilevel_subset_simulation(tol, L=5, gamma=0.5, y_L=-3.8):
    y = [-1.3, -2, -2.8, -3.3, y_L]
    cost = np.zeros(L)
    G_l = np.array([])
    N_l = 0
    err = 10  # Initial large error
    p_hat = 0

    # Level 1
    while err > tol:
        G = np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        G_l = np.append(G_l, G + kappa * gamma)
        mask = G_l <= y[0]
        p_hat = mask.mean()

        N_l += 1
        cost[0] += gamma ** (-2)
        
        if p_hat == 0 or p_hat == 1:
            err = 10
        else:
            err = p_hat * (1 - p_hat) / N_l
            
        # print(f"Level 1 - Iteration {N_l}: p_hat={p_hat:.3f}, err={err:.3e}")

    p_f = p_hat
    # print(f"Level 1 - Iteration {N_l}: p_hat={p_hat:.3f}, err={err:.3e}")

    # Levels 2 to L
    for l in range(1, L):
        G_l = G_l[mask][:1]
        err = 1e6  # Reset error
        while err > tol:
            G_new = 0.8 * G_l[-1] + np.sqrt(1 - 0.8**2) * np.random.normal(0, 1)
            kappa_new = np.random.uniform(-1, 1)
            G_l_new = G_new + kappa_new * gamma ** (l + 1)

            G_l = np.append(G_l, G_l_new if G_l_new <= y[l - 1] else G_l[-1])

            mask = G_l <= y[l]
            p_hat = mask.mean()

            err = delta(p_hat, mask)

            cost[l] += gamma ** (-2 * (l + 1))
            # print(f"Level {l + 1} - Iteration {len(G_l)}: p_hat={p_hat:.3f}, err={err:.3e}")
            # time.sleep(0.2)
            
            # if len(G_l) % 1000 == 0:
            #     print(f"Level {l + 1} - Iteration {len(G_l)}: p_hat={p_hat:.3f}, err={err:.3e}")

        p_f *= p_hat
        # print(f"Level {l + 1} - Iteration {len(G_l)}: p_hat={p_hat:.3f}, err={err:.3e}")

    # print(f"Final failure probability: {p_f:.3e}, rRMSE: {rRMSE([p_f]):.3e}\n")
    return p_f, cost


def run_simulation(args):
    tol, seed = args
    np.random.seed(seed)
    p_f, cost = adaptive_multilevel_subset_simulation(tol)
    return p_f, np.sum(cost)


if __name__ == "__main__":
    import multiprocessing
    import tqdm

    np.random.seed(42)
    error_list = []
    cost_list = []

    for tol in [0.1, 0.05, 0.03, 0.01]:
        failure_probabilities = []
        total_cost = []
        seeds = np.random.randint(10, 1000, 2)

        with multiprocessing.Pool(1) as pool:
            for p_f, cost in tqdm.tqdm(
                pool.imap(run_simulation, [(tol, seed) for seed in seeds]), total=2,
                desc=f"tol = {tol:.2f}"
            ):
                failure_probabilities.append(p_f)
                total_cost.append(np.sum(cost))

        err = rRMSE(failure_probabilities)
        error_list.append(err)
        cost = np.mean(total_cost)
        cost_list.append(cost)

        print(f"tol={tol:.2f}, error={err:.2e}, cost={cost:.2e}")

        np.save("AMLSS_error_list", error_list)
        np.save("AMLSS_cost_list", cost_list)

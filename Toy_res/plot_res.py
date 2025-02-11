import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Subset simulation
error_list_ss = np.load("SubS_error_list.npy")
cost_list_ss = np.load("SubS_cost_list.npy")

plt.loglog(error_list_ss, cost_list_ss, "tab:blue", label="Subset Simulation", marker='o')

x = np.linspace(1.1*error_list_ss[0], 0.9*error_list_ss[-1], 100)
y = 0.05 * cost_list_ss[0] * x ** (-4)

plt.loglog(x, y, color="tab:blue", linestyle='-.', label=r"$\epsilon^{-4}$")

# Subset simulation selective refinement
error_list_ss_sr = np.load("SubS_sr_error_list.npy")
cost_list_ss_sr = np.load("SubS_sr_cost_list.npy")

plt.loglog(error_list_ss_sr, cost_list_ss_sr, "tab:orange", label="Subset Simulation with Selective Refinement", marker='o')

x = np.linspace(1.1*error_list_ss_sr[0], 0.9*error_list_ss_sr[-1], 100)
y = 0.2 * cost_list_ss_sr[0] * x ** (-3)

plt.loglog(x, y, color="tab:orange", linestyle='-.', label=r"$\epsilon^{-3}$")

# error_list_ss_sr2 = np.load("SubS_sr_error_list2.npy")
# cost_list_ss_sr2 = np.load("SubS_sr_cost_list2.npy")

# plt.loglog(error_list_ss_sr2, cost_list_ss_sr2, "tab:purple", label="Subset Simulation with Selective Refinement", marker='o')

# x = np.linspace(1.1*error_list_ss_sr2[0], 0.9*error_list_ss_sr2[-1], 100)
# y = 0.1 * cost_list_ss_sr2[1] * x ** (-3)

# plt.loglog(x, y, color="tab:purple", linestyle='-.', label=r"$\epsilon^{-3}$")

# AMLSS
error_list_amlss = np.load("AMLSS_error_list.npy")
cost_list_amlss = np.load("AMLSS_cost_list.npy")

plt.loglog(error_list_amlss, cost_list_amlss, "tab:brown", label="AMLSS", marker='o')

x = np.linspace(1.1*error_list_amlss[0], 0.9*error_list_amlss[-1], 100)
y = 0.1 * cost_list_amlss[0] * x ** (-3)

plt.loglog(x, y, color="tab:brown", linestyle='-.', label=r"$\epsilon^{-3}$")

# MLE
error_list_mle = np.load("MLE_error.npy")
cost_list_mle = np.load("MLE_cost.npy")

plt.loglog(error_list_mle, cost_list_mle, "tab:green", label="MLE", marker='o')

x = np.linspace(1.1*error_list_mle[0], 0.9*error_list_mle[-1], 100)
y = 0.06 * cost_list_mle[0] * x ** (-2)

plt.loglog(x, y, color="tab:green", linestyle='-.', label=r"$\epsilon^{-2}$")

# MLE selective refinement
error_list_mle_sr = np.load("MLE_sr_error_list.npy")
cost_list_mle_sr = np.load("MLE_sr_cost.npy")

plt.loglog(error_list_mle_sr, cost_list_mle_sr, "tab:red", label="MLE with Selective Refinement", marker='o')

x = np.linspace(1.1*error_list_mle_sr[0], 0.9*error_list_mle_sr[-1], 100)
y = cost_list_mle_sr[0] * x ** (-2)

plt.loglog(x, y, color="tab:red", linestyle='-.', label=r"$\epsilon^{-2}$")

plt.xlabel("Relative Error")
plt.ylabel("Cost")
plt.legend()
plt.show()
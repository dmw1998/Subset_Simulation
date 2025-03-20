import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

fig,axs = plt.subplots(3,2,figsize=(15, 15))
fig.delaxes(axs[2,1])

# Data from PDE_res/SubS_numba.py
error_list = np.load("SubS_error_list512.npy")
cost_list = np.load("SubS_cost_list512.npy")
axs[0,0].loglog(error_list, cost_list, "o-", color="tab:blue", label="Data")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list), np.log(cost_list))
axs[0,0].loglog(error_list, np.exp(intercept) * error_list ** slope, "--", color="tab:blue", label=f"Fit: $\\epsilon^{{{slope:.2f}}}$")

# Plot the reference line
x = np.linspace(1.1*error_list[0], 0.9*error_list[-1], 100)
y = 0.75 * cost_list[0] * x ** (-2.6622)
axs[0,0].loglog(x, y, "--", color="fuchsia", label=r"$\epsilon^{-2.6622}$")

axs[0,0].set_title("Subset Simulation")
axs[0,0].legend()

# Data from PDE_res/SubS_sr.py
error_list_sr = np.load("SubS_sr_error_list512.npy")
cost_list_sr = np.load("SubS_sr_cost_list512.npy")
axs[0,1].loglog(error_list_sr, cost_list_sr, "o-", color = "tab:purple", label="Data")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_sr), np.log(cost_list_sr))
axs[0,1].loglog(error_list_sr, np.exp(intercept) * error_list_sr ** slope, "--", color="tab:purple", label=f"Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_sr[0], 0.9*error_list_sr[-1], 100)
y = cost_list_sr[0] * x ** (-2)
axs[0,1].loglog(x, y, "--", color = "fuchsia", label=r"$\epsilon^{-2}$")

axs[0,1].set_title("Subset Simulation with Selective Refinement")
axs[0,1].legend()

# Data from PDE_res/AMLSuS.py
error_list_amlsus = np.load("amlsus_error_list64.npy")
cost_list_amlsus = np.load("amlsus_cost_list64.npy")
axs[1,0].loglog(error_list_amlsus, cost_list_amlsus, "o-", color = "tab:orange", label="Data")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_amlsus), np.log(cost_list_amlsus))
axs[1,0].loglog(error_list_amlsus, np.exp(intercept) * error_list_amlsus ** slope, "--", color="tab:orange", label=f"Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_amlsus[0], 0.9*error_list_amlsus[-1], 100)
y = 0.7 * cost_list_amlsus[0] * x ** (-2)
axs[1,0].loglog(x, y, "--", color = "fuchsia", label=r"$\epsilon^{-2}$")

axs[1,0].set_title("Adaptive Multi-Level Subset Simulation")
axs[1,0].legend()

# Data from PDE_res/MLE.py
error_list_mle = np.load("mle_error_list64.npy")
cost_list_mle = np.load("mle_cost_list64.npy")
axs[1,1].loglog(error_list_mle, cost_list_mle, "o-", color = "tab:green", label="Data")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_mle), np.log(cost_list_mle))
axs[1,1].loglog(error_list_mle, np.exp(intercept) * error_list_mle ** slope, "--", color="tab:green", label=f"Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_mle[0], 0.9*error_list_mle[-1], 100)
y = 0.6 * cost_list_mle[0] * x ** (-2.6622)
axs[1,1].loglog(x, y, "--", color = "fuchsia", label=r"$\epsilon^{-2.6622}$")

axs[1,1].set_title("Multi-Level Estimator")
axs[1,1].legend()

# Data from PDE_res/MLE_sr.py
error_list_mle_sr = np.load("mle_sr_error_list64.npy")
cost_list_mle_sr = np.load("mle_sr_cost_list64.npy")
axs[2,0].loglog(error_list_mle_sr, cost_list_mle_sr, "o-", color = "tab:red", label="Data")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_mle_sr), np.log(cost_list_mle_sr))
axs[2,0].loglog(error_list_mle_sr, np.exp(intercept) * error_list_mle_sr ** slope, "--", color="tab:red", label=f"Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_mle_sr[0], 0.9*error_list_mle_sr[-1], 100)
y = 0.6 * cost_list_mle_sr[0] * x ** (-2)
axs[2,0].loglog(x, y, "--", color = "fuchsia", label=r"$\epsilon^{-2}$")

axs[2,0].set_title("MLE with Selective Refinement")
axs[2,0].legend()


# plt.xlabel("rRMSE")
# plt.ylabel("Cost")
plt.legend()
plt.savefig("1d_sub.png")
plt.show()
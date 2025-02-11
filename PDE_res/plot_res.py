import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.figure(figsize=(12, 8))

# Data from PDE_res/SubS_numba.py
error_list = np.load("SubS_error_list512.npy")
cost_list = np.load("SubS_cost_list512.npy")
plt.loglog(error_list, cost_list, "o-", color="tab:blue", label="Subset Simulation")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list), np.log(cost_list))
plt.loglog(error_list, np.exp(intercept) * error_list ** slope, "--", color="tab:blue", label=f"Subset Simulation Fit: $\\epsilon^{{{slope:.2f}}}$")

# Plot the reference line
x = np.linspace(1.1*error_list[0], 0.9*error_list[-1], 100)
y = 0.7 * cost_list[0] * x ** (-4)
plt.loglog(x, y, "--", color="tab:blue", label=r"$\epsilon^{-4}$")

# Data from PDE_res/SubS_sr.py
error_list_sr = np.load("SubS_sr_error_list512.npy")
cost_list_sr = np.load("SubS_sr_cost_list512.npy")
plt.loglog(error_list_sr, cost_list_sr, "o-", color = "tab:purple", label="Subset Simulation (SR)")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_sr), np.log(cost_list_sr))
plt.loglog(error_list_sr, np.exp(intercept) * error_list_sr ** slope, "--", color="tab:purple", label=f"Subset Simulation (SR) Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_sr[0], 0.9*error_list_sr[-1], 100)
y = 0.7 * cost_list_sr[0] * x ** (-3)
plt.loglog(x, y, "--", color = "tab:purple", label=r"$\epsilon^{-3}$")

# Data from PDE_res/AMLSuS.py
error_list_amlsus = [0.8, 0.6, 0.4]
cost_list_amlsus = [2.47e+12, 3.15e+12, 6.13e+12]
plt.loglog(error_list_amlsus, cost_list_amlsus, "o-", color = "tab:orange", label="AMLSuS")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_amlsus), np.log(cost_list_amlsus))
plt.loglog(error_list_amlsus, np.exp(intercept) * error_list_amlsus ** slope, "--", color="tab:orange", label=f"AMLSuS Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_amlsus[0], 0.9*error_list_amlsus[-1], 100)
y = 0.7 * cost_list_amlsus[0] * x ** (-2)
plt.loglog(x, y, "--", color = "tab:orange", label=r"$\epsilon^{-2}$")

# Data from PDE_res/MLE.py
error_list_mle = np.load("mle_error_list128.npy")
cost_list_mle = np.load("mle_cost_list128.npy")
plt.loglog(error_list_mle, cost_list_mle, "o-", color = "tab:green", label="MLE")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_mle), np.log(cost_list_mle))
plt.loglog(error_list_mle, np.exp(intercept) * error_list_mle ** slope, "--", color="tab:green", label=f"MLE Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_mle[0], 0.9*error_list_mle[-1], 100)
y = 0.7 * cost_list_mle[0] * x ** (-2)
plt.loglog(x, y, "--", color = "tab:green", label=r"$\epsilon^{-2}$")

# Data from PDE_res/MLE_sr.py
error_list_mle_sr = np.load("mle_sr_error_list128.npy")
cost_list_mle_sr = np.load("mle_sr_cost_list128.npy")
plt.loglog(error_list_mle_sr, cost_list_mle_sr, "o-", color = "tab:red", label="MLE (SR)")

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_mle_sr), np.log(cost_list_mle_sr))
plt.loglog(error_list_mle_sr, np.exp(intercept) * error_list_mle_sr ** slope, "--", color="tab:red", label=f"MLE (SR) Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.1*error_list_mle_sr[0], 0.9*error_list_mle_sr[-1], 100)
y = 0.7 * cost_list_mle_sr[0] * x ** (-2)
plt.loglog(x, y, "--", color = "tab:red", label=r"$\epsilon^{-2}$")



plt.xlabel("rRMSE")
plt.ylabel("Cost")
plt.legend()
plt.show()
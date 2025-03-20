import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.figure(figsize=(12, 8))
q = 0.6622
x = np.linspace(2.85e-1, 0.9, 100)

# # Data from PDE_res/SubS.py
# error_list = np.load("SubS_error_list512.npy")
# cost_list = np.load("SubS_cost_list512.npy")
# # plt.loglog(error_list, cost_list, "o-", color="tab:blue", label="SuS")

# # Linear Regression
# slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list), np.log(cost_list))
# plt.loglog(x, np.exp(intercept) * x ** slope, "--", color="tab:blue", label=f"SuS Fit: $\\epsilon^{{{slope:.2f}}}$")

# # # Plot the reference line
# # x = np.linspace(1.1*error_list[0], 0.9*error_list[-1], 100)
# # y = 0.75 * cost_list[0] * x ** (-2-q)
# # plt.loglog(x, y, "--", color="tab:blue", label=r"$\epsilon^{-2.6622}$")

# # Data from PDE_res/SubS_sr.py
# error_list_sr = np.load("SubS_sr_error_list512.npy")
# cost_list_sr = np.load("SubS_sr_cost_list512.npy")
# # plt.loglog(error_list_sr, cost_list_sr, "o-", color = "tab:purple", label="SuS-SR")

# # Linear Regression
# slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_sr), np.log(cost_list_sr))
# plt.loglog(x, np.exp(intercept) * x ** slope, "--", color="tab:purple", label=f"SuS-SR Fit: $\\epsilon^{{{slope:.2f}}}$")

# # x = np.linspace(1.1*error_list_sr[0], 0.9*error_list_sr[-1], 100)
# # y = cost_list_sr[0] * x ** (-2)
# # plt.loglog(x, y, "--", color = "tab:purple", label=r"$\epsilon^{-2}$")

# # Data from PDE_res/AMLSuS.py
# error_list_amlsus = np.load("amlsus_error_list64.npy")
# # error_list_amlsus = [9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1]
# cost_list_amlsus = np.load("amlsus_cost_list64.npy")
# # plt.loglog(error_list_amlsus, cost_list_amlsus, "o-", color = "tab:orange", label="AMLSuS")

# # Linear Regression
# slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_amlsus), np.log(cost_list_amlsus))
# plt.loglog(x, np.exp(intercept) * x ** slope, "--", color="tab:orange", label=f"AMLSuS Fit: $\\epsilon^{{{slope:.2f}}}$")

# # x = np.linspace(1.1*error_list_amlsus[0], 0.9*error_list_amlsus[-1], 100)
# # y = 0.7 * cost_list_amlsus[0] * x ** (-2)
# # plt.loglog(x, y, "--", color = "tab:orange", label=r"$\epsilon^{-2}$")

# Data from PDE_res/MLE.py
error_list_mle = np.load("mle_error_list64.npy")
cost_list_mle = np.load("mle_cost_list64.npy")
plt.loglog(error_list_mle, cost_list_mle, "o-", color = "tab:green", label="MLE")

# # Linear Regression
# slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_mle), np.log(cost_list_mle))
# plt.loglog(x, np.exp(intercept) * x ** slope, "--", color="tab:green", label=f"MLE Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(1.05*error_list_mle[0], 0.95*error_list_mle[-1], 100)
y = cost_list_mle[0] * x ** (-2.6622)
plt.loglog(x, y, "--", color = "tab:green", label=r"$\epsilon^{-2.6622}$")

# Data from PDE_res/MLE_sr.py
error_list_mle_sr = np.load("mle_sr_error_list64.npy")
cost_list_mle_sr = np.load("mle_sr_cost_list64.npy")
plt.loglog(error_list_mle_sr, cost_list_mle_sr, "o-", color = "tab:red", label="MLE-SR")

# Linear Regression
# slope, intercept, r_value, p_value, std_err = linregress(np.log(error_list_mle_sr), np.log(cost_list_mle_sr))
# plt.loglog(x, np.exp(intercept) * x ** slope, "--", color="tab:red", label=f"MLE-SR Fit: $\\epsilon^{{{slope:.2f}}}$")

x = np.linspace(error_list_mle_sr[0], 0.9*error_list_mle_sr[-1], 100)
y = cost_list_mle_sr[0] * x ** (-2)
plt.loglog(x, y, "--", color = "tab:red", label=r"$\epsilon^{-2}$")

plt.xlim(2.3e-1, 1.1)
plt.ylim(4e4, 2e9)
plt.xlabel("rRMSE")
plt.ylabel("Cost")
plt.legend()
# plt.grid()
plt.savefig("complexity_MLE.png")
# plt.savefig("complexity_SuS.png")
# plt.savefig("complexity_lr.png")
plt.show()
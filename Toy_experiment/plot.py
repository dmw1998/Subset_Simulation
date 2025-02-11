# """ MLW_sr.py results:
# N:  500
# Time:  18.31296420097351
# The mean cost:  6444.5
# rRMSE:  0.382511403641432

# N:  1000
# Time:  39.48930501937866
# The mean cost:  13630.0
# rRMSE:  0.2565747316233734

# N:  5000
# Time:  409.39933943748474
# The mean cost:  70715.0
# rRMSE:  0.11308309851958524

# N:  10000
# Time:  1603.2839212417603
# The mean cost:  142000.0
# rRMSE:  0.0921778958333096
# """
import numpy as np
import matplotlib.pyplot as plt

# # cost = [6444.5, 13630.0, 70715.0, 142000.0]
# # rrmse = [0.382511403641432, 0.2565747316233734, 0.11308309851958524, 0.09217789583330968]

# # plt.plot(rrmse, cost, 'o-')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel('relative error')
# # plt.ylabel('Cost')
# # plt.show()

# plt.figure(figsize=(8, 6))
# plt.xlim(0.01, 1)

# # Classical subset simulation
# cost_ss = [5.46e+05, 1.09e+06, 5.46e+06, 1.09e+07, 5.46e+07, 1.09e+08]
# err_ss = [7.17e-01 / 3, 4.88e-01 / 3, 3.28e-01 / 3, 3.01e-01 / 3, 2.8e-01 / 3, 1.7e-01 / 3]

# x = np.linspace(0.05, 0.3, 100)
# theoretical_ss = 3000 * np.array(x) ** (-4)

# plt.loglog(err_ss, cost_ss, 'bo-', label='Classical SubS')
# plt.plot(x, theoretical_ss, 'b--', label=r'O($\epsilon^{-4}$)')

# # Subset simulation with selective refinement
# cost_ss_sr = [1.36e+05 * 3, 2.73e+05 * 3, 4.09e+05 * 3, 1.36e+06 * 3, 2.73e+06 * 3, 4.09e+06 * 3, 4.09e+07 / 3, 1.36e+08 / 3]
# err_ss_sr = [4.81e-01 / 3, 4.54e-01 / 3, 3.83e-01 / 3, 2.55e-01 / 3, 2.35e-01 / 3, 2.19e-01 / 3, 2.15e-01 / 3, 1.95e-01 / 5]

# plt.loglog(err_ss_sr, cost_ss_sr, 'go-', label='SubS with SR')

# x = np.linspace(0.02, 0.2, 100)
# plt.plot(x, 8000 * x ** (-2) * np.log(1/x), 'g--', label=r'O($\epsilon^{-2}log(1 / \epsilon)$)')

# # Adaptive multilevel subset simulation
# cost_mlw_sr = [1.37e+06, 5.41e+06, 1.34e+08, 2.68e+08, 6.44e+09]
# err_mlw_sr = [8.89e-01, 3.07e-01, 2.20e-01, 9.22e-02, 7.32e-02]

# plt.loglog(err_mlw_sr, cost_mlw_sr, 'yo-', label='adpative ML')

# x = np.linspace(0.02, 1, 100)
# plt.plot(x, 500000 * x ** (-2), 'y--', label=r'O($\epsilon^{-2}$)')

# # Multilevel estimator
# cost_mle = [2.73e+07, 5.46e+07, 8.18e+07, 1.91e+08, 3.82e+08, 5.46e+09]
# err_mle = [3.68e-01, 3.08e-01, 2.61e-01, 1.88e-01, 1.82e-01/2, 1.52e-01/3]

# plt.loglog(err_mle, cost_mle, 'mo-', label='MLE')

# x = np.linspace(0.02, 0.4, 100)
# plt.plot(x, 2000000 * x ** (-2), 'm--', label=r'O($\epsilon^{-2}$)')

# # MLE_sr
# # Values of N used in the original code
# N_values = [80, 100, 500, 1000, 3000, 5000, 10000, 50000, 100000]

# # Initialize empty lists for error and cost
# err_list = []
# cost_list = []

# # Loop through each N to load the saved files and populate the lists
# for N in N_values:
#     # Load cost and error data for each N
#     costs = np.load(f"MLE_sr_costs_N_{N}.npy")
#     err = np.load(f"MLE_sr_relative_error_N_{N}.npy")
    
#     # Compute the mean of costs and add it to the cost_list
#     cost_list.append(np.mean(costs))
#     # Add the error to the err_list
#     err_list.append(err)
    
# plt.loglog(err_list, cost_list, 'ro-', label='MLE_sr')

# x = np.linspace(0.02, 0.8, 100)
# plt.plot(x, 30000 * x ** (-2), 'r--', label=r'O($\epsilon^{-2}$)')


# plt.xlabel('relative error')
# plt.ylabel('cost')
# plt.title('Comutational Complexity')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Data for different simulations
cost_ss = [5.46e+05, 1.09e+06, 5.46e+06, 1.09e+07, 5.46e+07, 1.09e+08]
err_ss = [7.17e-01 / 3, 4.88e-01 / 3, 3.28e-01 / 3, 3.01e-01 / 3, 2.8e-01 / 3, 1.7e-01 / 3]

cost_ss_sr = [1.36e+05 * 3, 2.73e+05 * 3, 4.09e+05 * 3, 1.36e+06 * 3, 2.73e+06 * 3, 4.09e+06 * 3, 4.09e+07 / 3, 1.36e+08 / 3]
err_ss_sr = [4.81e-01 / 3, 4.54e-01 / 3, 3.83e-01 / 3, 2.55e-01 / 3, 2.35e-01 / 3, 2.19e-01 / 3, 2.15e-01 / 3, 1.95e-01 / 5]

cost_mlw_sr = [1.37e+06, 5.41e+06, 1.34e+08, 2.68e+08, 6.44e+09]
err_mlw_sr = [8.89e-01, 3.07e-01, 2.20e-01, 9.22e-02, 7.32e-02]

cost_mle = [2.73e+07, 5.46e+07, 8.18e+07, 1.91e+08, 3.82e+08, 5.46e+09]
err_mle = [3.68e-01, 3.08e-01, 2.61e-01, 1.88e-01, 1.82e-01/2, 1.52e-01/3]

# Load data for MLE_sr
N_values = [80, 100, 500, 1000, 3000, 5000, 10000, 50000, 100000]
err_list = []
cost_list = []
for N in N_values:
    costs = np.load(f"MLE_sr_costs_N_{N}.npy")
    err = np.load(f"MLE_sr_relative_error_N_{N}.npy")
    cost_list.append(np.mean(costs))
    err_list.append(err)

# Plot settings
fig, axs = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1]})
fig.suptitle('Computational Complexity for Different Simulations', fontsize=16)

# Classical Subset Simulation and Subset Simulation with SR (Shared y-axis and x-axis)
axs[0, 0].loglog(err_ss, cost_ss, 'bo-', label='Classical SubS')
x = np.linspace(0.05, 0.3, 100)
axs[0, 0].plot(x, 3000 * x ** (-4), 'b--', label=r'O($\epsilon^{-4}$)')
axs[0, 0].set_title('Classical Subset Simulation')

axs[0, 1].loglog(err_ss_sr, cost_ss_sr, 'go-', label='SubS with SR')
x = np.linspace(0.02, 0.2, 100)
axs[0, 1].plot(x, 8000 * x ** (-2) * np.log(1/x), 'g--', label=r'O($\epsilon^{-2} \log(\epsilon)$)')
axs[0, 1].set_title('Subset Simulation with SR')

# Multilevel Estimator and MLE_sr (Shared y-axis and x-axis)
axs[1, 0].loglog(err_mle, cost_mle, 'mo-', label='MLE')
x = np.linspace(0.02, 0.4, 100)
axs[1, 0].plot(x, 2000000 * x ** (-2), 'm--', label=r'O($\epsilon^{-2}$)')
axs[1, 0].set_title('Multilevel Estimator')

axs[1, 1].loglog(err_list, cost_list, 'ro-', label='MLE_sr')
x = np.linspace(0.02, 0.8, 100)
axs[1, 1].plot(x, 30000 * x ** (-2), 'r--', label=r'O($\epsilon^{-2}$)')
axs[1, 1].set_title('MLE_sr')

# Adaptive Multilevel Subset Simulation with its own x-axis
axs[2, 0].loglog(err_mlw_sr, cost_mlw_sr, 'yo-', label='Adaptive ML')
x = np.linspace(0.02, 1, 100)
axs[2, 0].plot(x, 500000 * x ** (-2), 'y--', label=r'O($\epsilon^{-2}$)')
axs[2, 0].set_title('Adaptive Multilevel Subset Simulation')

# Remove the empty subplot in the third row, second column
fig.delaxes(axs[2, 1])

# Set shared y-axis for SS/SS_sr and MLE/MLE_sr, and shared x-axis for all subplots
axs[0, 0].sharey(axs[0, 1])
axs[1, 0].sharey(axs[1, 1])
for ax in axs.flat:
    ax.set(xlabel='Relative Error', ylabel='Cost')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

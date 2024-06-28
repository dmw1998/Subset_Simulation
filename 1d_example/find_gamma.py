import numpy as np
from scipy.optimize import fsolve

# Given values
y_values = [0.021068347431547174, 0.013600918999687628, 0.0046197026089659365, 10e-6]

# Define the system of equations based on the function definition
def equations(gamma):
    gamma = gamma[0]
    y_L = y_values[-1]
    L = len(y_values)
    y = np.zeros(L)
    y[-1] = y_L
    for i in range(L-1):
        l = L - 2 - i
        y[l] = (gamma ** (l+1) + gamma ** (l+2)) + y[l+1]
    return y - y_values

# Initial guess for gamma
gamma_guess = 0.5

# Solve for gamma
gamma_solution = fsolve(equations, gamma_guess)

print(f"Gamma: {gamma_solution[0]}")


def y_l(L, gamma):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    
    # output:
    # y: a sequence of the threshold values
    
    y = np.zeros(L)
    for i in range(L-1):
        l = L - 2 - i
        y[l] = (gamma ** l + gamma ** (l+1)) + y[l+1]
        
    return y

y = y_l(4, gamma_solution[0])
print(f"y: {y_values}")
print(f"y: {y}")
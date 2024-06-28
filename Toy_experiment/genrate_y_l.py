# Choose the threshold values
# y_l = (\gamma^{l} + \gamma^{l+1}) + y_{l+1}

import numpy as np

def y_l(gamma, y_L, L):
    # input:
    # gamma: (0,1)
    # y_L: the value of y_L
    # L: the number of layers
    
    # output:
    # y_l: the value of y_l
    
    y_l = np.zeros(L)
    y_l[L-1] = y_L
    
    for i in range(L-1):
        l = L - 2 - i
        y_l[l] = (gamma ** l + gamma ** (l+1)) + y_l[l+1]
        
    return y_l

if __name__ == "__main__":
    gamma = 0.5
    y_L = -3.8
    L = 5
    
    y = y_l(gamma, y_L, L)
    print("gamma: ", gamma)
    print("y: ", y)
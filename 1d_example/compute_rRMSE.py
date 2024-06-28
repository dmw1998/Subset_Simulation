# To compute the rRMSE

import numpy as np

def rRMSE(p_l, phi_l, N_l):
    # input:
    # p_l:= P(F_{l}|F_{l-1})
    # phi_l: autocorrelation factor
    # N_l: number of samples at level l
    
    # output:
    # err: relative root mean square error
    
    return np.sqrt( (1-p_l) * (1+phi_l) / p_l / N_l )
import numpy as np
from fenics import *

# Apply the subset simulation method with selective refinement strategy to the PDE model.

# Generate the samples for the Gaussian random field.
# We need to write a new method. Combine the subset simulation and the selective refinement strategy.

# Solving the 1d poisson equation with the finite element method.
def solving_pde(theta, l):
    # input:
    # theta: the parameter of the Gaussian random field
    # l: the level -- related to mesh size

    # output:
    # u_h: the solution of the PDE

    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"

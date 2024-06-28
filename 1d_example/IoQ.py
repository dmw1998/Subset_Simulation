# To compute the interests of quantities.

from fenics import *
from kl_expansion import *
import numpy as np
import matplotlib.pyplot as plt

def IoQ(a_x, n_grid):
    # input:
    # a_x: the random field (coefficients)
    # n_grid: number of mesh points
    
    # output:
    # u_h(1): approximated IoQ
    
    # Set PETSc options to suppress solver output
    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    
    # Create the mesh and define function space
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    # Define the random field a(x) on the FEniCS mesh
    a = Function(V)
    a_values = np.interp(mesh.coordinates().flatten(), np.linspace(0, 1, len(a_x)), a_x)
    a.vector()[:] = a_values
    
    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    bc = DirichletBC(V, u0, boundary)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    
    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Compute solution
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)  # Suppress the FEniCS log messages
    solve(a_form == L, u_h, bc)
    
    return u_h(1)

if __name__ == "__main__":
    # Define the number of terms in the KL expansion
    M = 150
    
    # Define the correlation length
    l_c = 0.1
    
    # Define the KL coefficients
    thetas = np.random.normal(0, 1, M)
    
    # Compute the random field
    a_x = kl_expan(thetas)
    
    # Compute the IoQ
    u_h = IoQ(a_x, 100)
    print(u_h)
    
    # Plot the random field
    x = np.linspace(0, 1, 1000)
    plt.plot(x, a_x)
    plt.show()
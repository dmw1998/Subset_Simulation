import numpy as np
from fenics import *

# Apply the subset simulation method with selective refinement strategy to the PDE model.

# Generate the samples for the Gaussian random field.
# We need to write a new method. Combine the subset simulation and the selective refinement strategy.

# Computing the coefficient for the pde
def kl_expan(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    # Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
    # def eigenvalue(m):
    #     return 0.02 / np.pi ** 2 / (m + 0.5) ** 2
    
    # def eigenfunction(m, x):
    #     return np.sqrt(2) * (np.sin((m + 0.5) * np.pi * x) + np.cos((m + 0.5) * np.pi * x))
    
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m*np.pi
        return 2*beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m*np.pi
        A = np.sqrt(2 * w**2 / (2*beta + w**2 + beta**2))
        B = np.sqrt(2 * beta**2 / (2*beta + w**2 + beta**2))
        return A*np.cos(w*x) + B*np.sin(w*x)
    
    # p = 0
    # for m in range(M):
    #     p += np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1,x)
    #     plt.plot(x, p)
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * thetas[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

# Solving the 1d poisson equation with the finite element method.
def solving_pde(theta, l):
    # input:
    # theta: the parameter of the Gaussian random field
    # l: the level -- related to mesh size

    # output:
    # u_h: the solution of the PDE

    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    
    n_grid = 64 * 2 ** l
    a_x = kl_expan(theta)
    
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
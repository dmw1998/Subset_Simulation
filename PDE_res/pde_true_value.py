from fenics import *
import numpy as np
import csv

def kl_expan(theta):
    M = np.size(theta)
    
    x = np.linspace(0, 1, 1000)
    
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m * np.pi
        return 2 * beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m * np.pi
        A = np.sqrt(2 * w**2 / (2*beta + w**2 + beta**2))
        B = np.sqrt(2 * beta**2 / (2*beta + w**2 + beta**2))
        return A*np.cos(w*x) + B*np.sin(w*x)
    
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * theta[m] for m in range(M))
    
    a_x = np.exp(log_a_x)
    
    return a_x

def IoQ(a_x, n_grid):
    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    a = Function(V)
    a_values = np.interp(mesh.coordinates().flatten(), np.linspace(0, 1, len(a_x)), a_x)
    a.vector()[:] = a_values
    
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    bc = DirichletBC(V, u0, boundary)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    
    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx
    
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L, u_h, bc)
    
    return u_h(1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    np.random.seed(6)
    levels = range(1, 12)
    n_grids = 2**np.array(levels)
    h = 1 / n_grids
    log_h = np.log(h)
    seeds = np.random.randint(0, 1000, 100)
    slopes = np.array([])
    intercepts = np.array([])
    
    for seed in seeds:
        # Compute the true value
        np.random.seed(seed)
        theta = np.random.normal(0, 1, 150)
        a_x = kl_expan(theta)
        u_h = IoQ(a_x, 1000000)
        true_value = u_h
        
        # Compute the errors
        errors = np.array([])
        for n_grid in n_grids:
            np.random.seed(seed)
            theta = np.random.normal(0, 1, 150)
            a_x = kl_expan(theta)
            u_h = IoQ(a_x, n_grid)
            error = np.sqrt((u_h - true_value)**2)
            errors = np.append(errors, error)
            
        # Compute the convergence rate
        log_errors = np.log(errors[6:-1])
        slope, intercept, r_value, p_value, std_err = linregress(log_h[6:-1], log_errors)
        # print("Slope: {:.2f}, C: {:.5f}".format(slope, np.exp(intercept)))
        slopes = np.append(slopes, slope)
        intercepts = np.append(intercepts, intercept)
        
        # Plot the errors
        plt.loglog(h, errors, 'o-', color = 'tab:purple', alpha=seed/2000)
            
    # Compute the average convergence rate
    avg_slope = np.mean(slopes)
    max_intercept = np.max(intercepts)
    print("Average slope: {:.2f}, C: {:.5f}".format(avg_slope, np.exp(max_intercept)))
        
    plt.plot(h[5:], np.exp(max_intercept) * h[5:]**avg_slope, 'b-', label='${:.2f} h^{{{:.2f}}}$'.format(np.exp(max_intercept), avg_slope))
    plt.loglog(h, h**2, '--', label='$h^2$')
    plt.loglog(h, h, '--', label='$h$')
    plt.xlabel('h')
    plt.ylabel('Errors')
    plt.grid()
    plt.legend()
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt

# Parameters
D  = 1.0       # diffusion coefficient
k2 = 0.1       # quadratic reaction rate
k3 = 0.01      # cubic reaction rate

L  = 1.0       # domain length
Nx = 100       # number of spatial points
dx = L / (Nx - 1)

dt = 1e-6      # time step (must satisfy stability cond ition)
Nt = 100      # number of time steps

# Spatial grid
x = np.linspace(0, L, Nx)

# Initial condition
def N0(x):
    return np.exp(-200*(x-0.5)**2)  # Gaussian bump

N = N0(x).copy()

# For stability of explicit diffusion:
# dt < dx^2/(2D)
print("Stability limit dt < ", dx**2/(2*D))

# Time stepping
for n in range(Nt):
    # Laplacian using central difference
    d2N_dx2 = (np.roll(N, -1) - 2*N + np.roll(N, 1)) / dx**2
    
    # Neumann BC (zero gradient): copy edges
    d2N_dx2[0]  = d2N_dx2[1]
    d2N_dx2[-1] = d2N_dx2[-2]
    
    # PDE update
    N = N + dt * (-D * d2N_dx2 - k2 * N**2 - k3 * N**3)
    N = np.clip(N, 0, 1e3)  # or whatever max is physically reasonable

# Plot final result
plt.plot(x, N0(x), label='Initial')
plt.plot(x, N, label='Final')
plt.legend()
plt.xlabel('x')
plt.ylabel('N(x)')
plt.show()

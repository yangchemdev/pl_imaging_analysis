import numpy as np
import matplotlib.pyplot as plt

# Parameters
D  = 1.0e-3       # diffusion coefficient in um2/ns. Equivalent to 0.1 * x cm2/s 
k2 = 0.1       # quadratic reaction rate, unit in 
k3 = 0      # cubic reaction rate

L  = 1.0       # domain length in um
Nx = 100       # number of spatial points
dx = L / (Nx - 1)

dt = 1e-6      # time step (must satisfy stability cond ition)
Nt = 100      # number of time steps

# Spatial gride
x = np.linspace(-L/2, L/2, Nx)  # grid, centered at x=0
n0_pop = 3.7e5  # Integrated n0 concentration unit in 1/um2. 
n0_sigma = 0.2  # width of Gaussian bump unit in um
# Initial condition
def area_gaussian(x, sigma, A, x0):
    return A * 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - x0)**2 / (2 * sigma**2))

def N0_x(x):
    return area_gaussian(x, n0_sigma, n0_pop, 0)

N = N0_x(x).copy()

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
plt.plot(x, N0_x(x), label='Initial')
plt.plot(x, N, label='Final')
plt.legend()
plt.xlabel('x')
plt.ylabel('N(x)')
plt.show()

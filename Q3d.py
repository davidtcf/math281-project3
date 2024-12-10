import numpy as np
import matplotlib as mp
import matplotlib.colors
import matplotlib.pyplot as plt
mp.rcParams['figure.dpi'] = 300

# Parameters
Lx = 10         # Domain length
nu = 0.1        # Diffusion coefficient
Nx = 1000        # Number of spatial grid points (adjustable)
T = 50          # Total simulation time
Nt = 100000       # Number of time steps (adjustable)

# Derived quantities
dx = Lx / Nx    # Spatial resolution
dt = T / Nt     # Time step
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)  # Spatial grid
t = np.linspace(0, T, Nt)  # Time array

# Stability check
if dt > (dx**2 / (2 * nu)):
    raise ValueError("Time step size does not satisfy the CFL condition for diffusion!")

# Initial condition
u = np.sin(4 * np.pi * x / Lx) + 0.25 * np.sin(8 * np.pi * x / Lx)

# Solution array to store results at each time step
solution = np.zeros((Nt, Nx))
solution[0, :] = u  # Store initial condition

# Time-stepping loop (Second-order Central Difference Scheme for Diffusion)
u_new = np.copy(u)
for n in range(1, Nt):
    # Update the interior points using the second derivative
    u_new[1:-1] = u[1:-1] + nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    
    # Apply periodic boundary conditions
    u_new[0] = u[0] + nu * dt / dx**2 * (u[1] - 2 * u[0] + u[-1])  # Left boundary
    u_new[-1] = u[-1] + nu * dt / dx**2 * (u[0] - 2 * u[-1] + u[-2])  # Right boundary

    # Store the result for the current time step
    solution[n, :] = u_new
    
    # Update for the next step
    u = np.copy(u_new)

# Visualization using pcolormesh
cv = np.max(np.abs(solution))  # Compute max absolute value for color scaling
plt.figure(figsize=(8, 6))
plt.pcolormesh(x, t, solution, norm=matplotlib.colors.SymLogNorm(linthresh=0.005, linscale=0.5, vmin=-cv, vmax=cv, base=10) ,shading='auto', cmap='RdBu_r')

# Add colorbar and labels
cbar = plt.colorbar()
cbar.ax.set_ylabel('u', fontsize=12)
plt.xlabel('x', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Solution of the Diffusion Equation (Color Mesh)', fontsize=14)
plt.show()

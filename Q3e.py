import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['figure.dpi'] = 300

# Parameters
Lx = 10                 # Length of the spatial domain
Nx = 100                # Number of spatial points
dx = Lx / Nx            # Spatial step size
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)  # Periodic grid
nu = 0.1                # Diffusion coefficient
dt = 0.01               # Time step size
T = 50                  # Total simulation time
Nt = int(T / dt)        # Number of time steps

# Stability condition for explicit method
assert dt <= dx**2 / (2 * nu), "Time step too large for stability."

# Initial condition
u = np.sin(4 * np.pi * x / Lx) + 0.25 * np.sin(8 * np.pi * x / Lx)

# Storage for visualization
time_snapshots = [0, int(Nt / 4), int(Nt / 2), int(3 * Nt / 4), Nt - 1]  # Adjusted snapshots
solutions = []
integrals = []  # To store the integral of u^2
time_points = []  # To store corresponding time points

# Time stepping loop
for n in range(Nt):
    # Apply periodic boundary conditions using modular indexing
    u_next = np.zeros_like(u)
    for i in range(Nx):
        u_next[i] = u[i] + nu * dt / dx**2 * (
            u[(i+1) % Nx] - 2 * u[i] + u[(i-1) % Nx]
        )
    u = u_next
    
    # Compute the integral of u^2 over the domain
    integral = np.trapz(u**2, x)  # Or use np.sum(u**2) * dx
    integrals.append(integral)
    time_points.append(n * dt)
    
    # Save solution at specific time steps for visualization
    if n in time_snapshots:
        solutions.append(u.copy())

plt.figure(figsize=(10, 6))
plt.plot(time_points, integrals, label=r"$E(t)=\int u^2 dx$", color='r')
plt.xlabel("Time")
plt.ylabel(r"$E(t)$")
plt.title("$E(t)$ plotted over Time")
plt.grid()
plt.legend()
plt.show()

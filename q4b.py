import numpy as np
import matplotlib.pyplot as plt

# Define the problem parameters
Nmodes = 1
fig = plt.figure(dpi=300)

# Functions for the chosen problem
def f(x):
    return np.sin(Nmodes * x)

def f_third_derivative(x):
    return -Nmodes**3 * np.cos(Nmodes * x)

Lx = 2 * np.pi
Nxs = np.power(2, np.arange(2, 9))

err_5 = np.zeros(Nxs.shape)

gridspec_props = dict(wspace=0.05, hspace=0.5, left=0.1, right=0.8, bottom=0.1, top=0.9)

for Nx, ind in zip(Nxs, range(len(Nxs))):
    # Grid with chosen resolution
    x = np.linspace(0, Lx, Nx)[:-1]
    Delta_x = x[1] - x[0]

    # Function to differentiate
    y = f(x)

    # True third derivative
    yp3 = f_third_derivative(x)

    # Compute the numerical third derivative (5th-order scheme)
    Ord5 = (-0.5 * np.roll(y, 2) + np.roll(y, 1) - np.roll(y, -1) + 0.5 * np.roll(y, -2)) / (Delta_x**3)

    # Store the error in the derivatives
    err_5[ind] = np.sqrt(np.mean((Ord5 - yp3) ** 2))


# Plot the error values
plt.figure(figsize=(10, 5))
plt.plot(Lx/Nxs, err_5, 'o-', label='centered, second-order scheme for 3rd derivative')
plt.plot(Lx/Nxs, (Lx/Nxs)**2, '-.k', label='$dx^2$')
plt.xlabel(r'$\Delta x$')
plt.ylabel('Finite Difference Error')
plt.xscale('log')
plt.yscale('log')
plt.title('Demostartion of Convergence')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('q4b.png')
plt.show()

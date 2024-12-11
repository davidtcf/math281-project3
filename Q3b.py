import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams['figure.dpi'] = 300

fig, axes = plt.subplots(1, 1, figsize=(6, 4))
fig.subplots_adjust(wspace=0.05, hspace=0.5, left=0.1, right=0.8, bottom=0.1, top=0.9)

def u(x):
    return np.sin(x)

def u_2prime(x):
    return -np.sin(x)
Lx = 2*np.pi
Nxs = np.power(2, np.arange(2, 9))
err = np.zeros(Nxs.shape)

for ind, Nx in enumerate(Nxs):
    x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    Delta_x = x[1] - x[0]

    y = u(x)
    yp = u_2prime(x)

    d2udx2 = (np.roll(y, 1) - 2 * y + np.roll(y, -1)) / (Delta_x**2)

    err[ind] = np.sqrt(np.mean((d2udx2 - yp) ** 2))

axes.plot(Lx / Nxs, err, '-o', label='2nd order')
axes.plot(Lx / Nxs, (Lx / Nxs)**2, '--k', label='$dx^2$ ')
axes.set_yscale('log')
axes.set_xscale('log')

axes.set_xlabel('$\Delta x$')
axes.set_ylabel('Finite Difference Error')
axes.set_title(f'Convergence Order for $\sin(x)$')
axes.legend()

# Show plot
plt.tight_layout()
plt.show()

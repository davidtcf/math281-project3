import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.fftpack import fft

# First, we need to load in the data
Burgers_data = np.load('Burgers.npz')

time = Burgers_data['time']         # The temporal grid
x    = Burgers_data['x']            # The spatial grid
u    = Burgers_data['u']            # The solution (first axis is time, second is space)
kx   = Burgers_data['kx']           # The wavenumbers relevant to the grid
nu   = float(Burgers_data['nu'])    # The viscosity coefficient

Nt, Nx = u.shape

fig, ax = plt.subplots(1, 1, figsize = (6,4))

uhat = fft(u, axis = 1)
power = np.abs(uhat)**2

# The [:Nx//2] keeps only the first half of the wavenumbers [the non-negative ones]
q0 = ax.pcolormesh( kx[:Nx//2], time, power[:,:Nx//2], cmap = 'plasma', norm = colors.LogNorm( vmin = 1e-10 ) ) # This sets the colour bar to use a log scale

plt.colorbar(q0, ax = ax)

plt.xscale('log')
plt.xlim( kx[1], kx[Nx//2-1] )

ax.set_xlabel('Wavenumber')
ax.set_ylabel('Time')
ax.set_title('Spectral Power of viscous Burgers equation')
plt.savefig('q5d1.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots(1, 1, figsize = (6,4))

v = 0.00078125
Renolds = [1,2,5]

for Re in Renolds:
    rms_u = np.sqrt(np.mean(u**2,axis=1))
    L = v*Re/rms_u
    plt.plot(time,L,label='Re = '+str(Re))

ax.set_xlabel('Time')
ax.set_ylabel(r'$L^*$')
ax.set_title(r'Characteristic length-scale $L^*$ vs Time for different Reynolds numbers')
plt.legend()
plt.savefig('q5d2.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

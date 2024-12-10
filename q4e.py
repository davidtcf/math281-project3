import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Final time of the simulation
final_time = 50.

# Number of points in the computational grid
Nx = 64

# CFL factor
CFL = 0.001

# PDE Parameters
U0 = 0.1
Lx = 10.

# dx and dt
delta_x = Lx / Nx
delta_t = CFL * delta_x**3 / U0


# Create spatial grid
x = np.arange(delta_x / 2. - Lx/2, Lx/2, delta_x)

# Specify the number of time points to store (Nouts)
#  they will be equally spaced throughout the simulation.
Nouts = 64
output_interval = final_time / Nouts
t = np.zeros(Nouts + 1)
t[0] = 0.

# Create an array to store the solution, and write in the initial conditions
solution = np.zeros((Nouts + 1,Nx))
#solution[0,:] = np.sin( 4 * np.pi * x / Lx )
solution[0,:] = np.sin( 4 * np.pi * x / Lx ) + 0.25 * np.sin( 8 * np.pi * x / Lx )
#solution[0,:] = 0.25 * np.sin( 8 * np.pi * x / Lx ) 

def ddx(f, dx = delta_x):
    
    dfdx = ( - 0.5 * np.roll(f, 2) + np.roll(f, 1) - np.roll(f, -1) + 0.5 * np.roll(f, -2)) / ( dx**3 )
    
    return dfdx

curr_time = t[0]
u = solution[0,:]

next_output_time = output_interval
output_number = 0
while curr_time < final_time:
    
    # Compute the RHS of the ODE
    dudt = U0 * ddx( u )
    
    # Update first-order record
    u = u + delta_t * dudt
    
    # Increase 'time' by Delta t
    curr_time = curr_time + delta_t
    
    # Store the new values in our array, if at the right time
    if curr_time >= next_output_time:
        output_number = output_number + 1
        next_output_time += output_interval
        
        solution[output_number,:] = u
        t[output_number] = curr_time
    
# Helps to avoid odd errors from funky step sizes
t = t[:output_number]
solution = solution[:output_number,:]
energy = np.sum(solution**2, axis = 1)
energy = energy.astype(int)

plt.plot(t, energy)
plt.xlabel('Time')
plt.ylabel(r'Energy:$E(t)=\int_x u^2(x, t) d x$')
plt.title(r'Energy of Numerical Solution Demostrate $\frac{d}{d t} E(t)=0$')
plt.tight_layout()
plt.savefig('q4e.png', dpi = 300, bbox_inches = 'tight')
plt.show()
plt.close()

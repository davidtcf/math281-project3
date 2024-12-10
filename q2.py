import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from matplotlib import cm

#Color Mesh

Lx = 10        
U0 = 0.1        
Nx = 100      
final_time = 50 
output_interval = 0.5  
delta_t = 0.01  
dx = Lx / Nx    
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False) 

if delta_t > dx / abs(U0):
    raise ValueError("Doesn't Satisfy CFL Condition")

u = np.sin(4 * np.pi * x / Lx) + 0.25 * np.sin(8 * np.pi * x / Lx)

def ddx(f, dx=dx):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)

output_number = 0
curr_time = 0.0
next_output_time = output_interval
num_outputs = int(final_time / output_interval) + 1
energy=np.zeros(num_outputs)
solution = np.zeros((num_outputs, Nx))
t = np.zeros(num_outputs)
solution[output_number, :] = u
t[output_number] = curr_time

while curr_time < final_time:

    dudt = -U0 * ddx(u, dx)
    
    u = u + delta_t * dudt
    
    curr_time += delta_t
    
    if curr_time >= next_output_time:
        output_number += 1
        next_output_time += output_interval
        
        solution[output_number, :] = u
        t[output_number] = curr_time
        energy[output_number]=np.sum(u**2)*dx

    u_initial = np.sin(4 * np.pi * x / Lx) + 0.25 * np.sin(8 * np.pi * x / Lx)
    energy[0]=np.sum(u_initial**2)*dx

solution = solution[:output_number + 1, :]
t = t[:output_number + 1]

cv = np.max(np.abs(solution)) 
plt.figure(figsize=(8, 6))
plt.pcolormesh(x, t, solution, shading='auto', cmap='coolwarm', vmin=-cv, vmax=cv)
cbar = plt.colorbar()
cbar.ax.set_ylabel('u', fontsize=12)
plt.xlabel('Position (m)', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.title('Time Evolution of the Advection PDE (Color Mesh, dt=0.005)', fontsize=14)
plt.plot([x[0],x[50]],[0,(x[50]-x[0])/U0], color="green")
plt.show()

#Spectral Power

coeff = np.zeros((len(t), Nx), dtype=complex)
for i in range(len(t)):
    coeff[i, :] = fft(solution[i, :])

kx = fftfreq(Nx, d=dx)

kx_pos = kx[:Nx // 2]

specpower = np.abs(coeff)**2

plotindices = np.arange(0, len(t), max(len(t) // 10, 1))

plt.figure(figsize=(8, 6))

cmap = plt.get_cmap('viridis')  
norm = plt.Normalize(vmin=min(t), vmax=max(t))  
sm = cm.ScalarMappable(cmap=cmap, norm=norm)  

for time in plotindices:
    color = cmap(norm(t[time]))  
    plt.plot(kx_pos, specpower[time, :Nx//2], color=color, label=f'Time {t[time]:.2f}')

cbar = plt.colorbar(sm, ax=plt.gca())
cbar.ax.set_ylabel('Time (s)', fontsize=12)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(Wavenumber)', fontsize=12)
plt.ylabel('log(Spectral Power)', fontsize=12)
plt.title('Power Spectrum for the Numerical Solution (log-log Space)', fontsize=14)
plt.grid(True)
plt.show()

#Spatial Plot of u(x,t)

time_indices = np.linspace(0, len(t) - 1, 5, dtype=int)

plt.figure(figsize=(10, 8))
for i, idx in enumerate(time_indices):
    plt.plot(x, solution[idx, :], label=f'Time {t[idx]:.2f}')
plt.xlabel('x', fontsize=12)
plt.ylabel('u(x, t)', fontsize=12)
plt.title('Spatial Plot of u(x,t) at Selected Times', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.show()

time_indices = np.linspace(0, len(t) - 1, 8, dtype=int)
cmap = plt.get_cmap('viridis')  
norm = plt.Normalize(vmin=min(t), vmax=max(t))  

plt.figure(figsize=(10, 8))

lines = []
ax = plt.gca()

for i, idx in enumerate(time_indices):
    line = plt.plot(x, solution[idx, :], label=f'Time {t[idx]:.2f}', color=cmap(norm(t[idx])))
    lines.append(line[0])

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.colorbar(sm, ax=ax, label='t (s)')
plt.xlabel('x (m)', fontsize=12)
plt.ylabel('u', fontsize=12)
plt.title('Spatial Plot of u(x,t) at Selected Times (dt=0.005)', fontsize=14)
plt.grid()
plt.show()

#RMS Error

def exact_solution(x, t, U0, Lx, initial_condition_func):
    return initial_condition_func(x - U0 * t, Lx)

def initial_condition_func(x, Lx):
    return np.sin(4 * np.pi * x / Lx) + 0.25 * np.sin(8 * np.pi * x / Lx)

def rms_error(numerical_solution, exact_solution, dx):
    return np.sqrt(np.mean((numerical_solution - exact_solution)**2))

rms_errors = []
for i in range(len(t)):
    u_exact_t = exact_solution(x, t[i], U0, Lx, initial_condition_func)  
    rms_err = rms_error(solution[i, :], u_exact_t, dx)  
    rms_errors.append(rms_err)

plt.figure(figsize=(8, 6))
plt.plot(t, rms_errors, label='RMS Error')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('RMS Error', fontsize=12)
plt.title('Root Mean Squared Error vs. Time', fontsize=14)
plt.legend()
plt.grid()
plt.show()

print(plotindices)

plt.figure(figsize=(8, 6))
plt.plot(t, energy, label='Energy Plot')
plt.ylim(0,10)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Energy (J)', fontsize=12)
plt.title('Energy vs. Time in Spatial Domain', fontsize=14)
plt.legend()
plt.grid()
plt.show()

print(energy)

#_____________________________________________________________________

Lx = 10         
U0 = 0.1        
Nx = 100       
final_time = 50 
output_interval = 0.5  
delta_t = 0.005  

dx = Lx / Nx    
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)  

if delta_t > dx / abs(U0):
    raise ValueError("Time step size does not satisfy the CFL condition!")

u2 = np.sin(4 * np.pi * x / Lx) + 0.25 * np.sin(8 * np.pi * x / Lx)

output_number = 0
curr_time = 0.0
next_output_time = output_interval

num_outputs = int(final_time / output_interval) + 1
solution2 = np.zeros((num_outputs, Nx))
t2 = np.zeros(num_outputs)

solution2[output_number, :] = u2
t2[output_number] = curr_time

while curr_time < final_time:
    dudt = -U0 * ddx(u2, dx)
    
    u2 = u2 + delta_t * dudt
    
    curr_time += delta_t
    
    if curr_time >= next_output_time:
        output_number += 1
        next_output_time += output_interval
        
        solution2[output_number, :] = u2
        t2[output_number] = curr_time

solution2 = solution2[:output_number + 1, :]
t2 = t2[:output_number + 1]

tolerance = 1e-1  

difference = np.zeros((len(t), len(x)))

for t1_idx, t1_val in enumerate(t):
    match_idx = np.where(np.abs(t2 - t1_val) < tolerance)[0]
    
    if match_idx.size > 0:
        index_in_solution2 = match_idx[0]
        
        difference[t1_idx, :] = solution[t1_idx, :] - solution2[index_in_solution2, :]
    else:
        print(f"Warning: No matching time step for t1 = {t1_val} in t2.")

print(f"Shape of difference array: {difference.shape}")
print(f"Number of non-zero entries: {np.count_nonzero(difference)}")

cv = np.max(np.abs(difference))  
plt.figure(figsize=(8, 6))
plt.pcolormesh(x, t, difference, shading='auto', cmap='coolwarm', vmin=-cv, vmax=cv)

cbar = plt.colorbar()
cbar.ax.set_ylabel('Difference', fontsize=12)
plt.xlabel('Position (m)', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.title('Time Evolution of the Difference (solution1 - solution2)', fontsize=14)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 10.0    # Length of the transmission line
Nx = 10000     # Number of spatial points
dx = Lx / Nx  # Space step
dt = 0.001e-9    # Time step
Nt = 10000     # Number of time steps

# Circuit Parameters
L = 250e-9  # Inductance per unit length
C = 100e-12   # Capacitance per unit length
R = 1  # Resistance per unit length
G = 1e-9   # Conductance per unit length

# Stability condition (CFL-like)
assert dt <= dx / np.sqrt(L * C), "Choose a smaller dt for stability!"

# Initialize V and I
V = np.zeros((Nx, Nt))  # Voltage array
I = np.zeros((Nx, Nt))  # Current array

# Initial condition: Voltage pulse in the middle
V[Nx//2, 0] = 1.0

# Finite Difference Time Evolution
for n in range(0, Nt - 1):
    for i in range(1, Nx - 1):  # Avoid boundary points
        # Update I using first equation
        I[i, n+1] = I[i, n] - (dt / L) * ((V[i+1, n] - V[i, n]) / dx + R * I[i, n])

        # Update V using second equation
        V[i, n+1] = V[i, n] - (dt / C) * ((I[i+1, n] - I[i, n]) / dx + G * V[i, n])

# Visualization
plt.imshow(V, extent=[0, Lx, 0, Nt*dt], aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label="Voltage")
plt.xlabel("x (Position)")
plt.ylabel("Time")
plt.title("Telegrapher's Equations Simulation")
plt.show()
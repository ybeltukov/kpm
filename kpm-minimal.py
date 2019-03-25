"""
This program shows how to calculate the vibrational density of states using
the Kernel Polynomial Method. The details of this method can be found in the
following article and the references therein:

Y. M. Beltukov, C. Fusco, D. A. Parshin, A. Tanguy,
Boson peak and Ioffe-Regel criterion in amorphous silicon-like materials: 
The effect of bond directionality. Physical Review E 93, 023006 (2016).
http://doi.org/10.1103/PhysRevE.93.023006

Please cite this article if you find it useful.

Yaroslav Beltukov
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.fftpack

# Generate some sparse dynamical matrix
A = sp.sparse.rand(5000, 10000, 0.001, random_state=0)
A.data -= 0.5
M = A@A.T

reps = 30 # a number of repetitions to average the result
omega_max = 1.9 # omega_max should be bigger than any frequency in the system
deg = 200 # polynomial degree, which gives the frequency resolution ~omega_max/deg
sub = 3 # a number of subpoints to get a smooth result

dt = 2/omega_max
m = np.zeros(sub*deg)
for i in range(reps):
    # the main loop has the Verlet form
    v0 = np.random.randn(M.shape[0])
    v0 /= np.linalg.norm(v0)
    u1, u0 = dt*v0, 0
    m[0] += u1@v0
    for k in range(1, deg):
        u1, u0 = 2*u1 - u0 - dt**2*M@u1, u1
        m[k] += u1@v0
m /= reps

# Apply damping to avoid Gibb's oscillations
def jackson(k, K):
    return ((K - k)*np.cos(np.pi*k/K) + np.sin(np.pi*k/K)/np.tan(np.pi/K))/K
m[:deg] *= jackson(1 + np.arange(deg), deg + 1)/jackson(1, deg + 1)

# Calculate resulting arrays
omega = omega_max*np.sin(np.pi/2*np.arange(0.5, sub*deg)/sub/deg)
g = 2/np.pi*omega/omega_max*sp.fftpack.dst(m, type=3)
# You can apply np.interp to resample the resulting VDOS

# Plot the resulting VDOS
plt.plot(omega, g)
plt.xlabel('Frequency')
plt.ylabel('VDOS')
# Compare with the histogram of eigenfrequencies
plt.hist(np.sqrt(np.abs(np.linalg.eigvalsh(M.toarray()))), 50, density=True)


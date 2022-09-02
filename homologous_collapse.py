#!/usr/bin/env python3
'''
Homologous collapse
du/dtau = -1. / x**2
dx/dtau = u
t=0, x=1, u=0

u = v / alpha / r0
x = r / r0
tau = alpha * t
alpha = np.sqrt(G * M / r0**3)

Carroll, Cap. 12, p. 415, Eq. 12.19
'''
import numpy as np
import matplotlib.pyplot as plt


G = 6.67259e-8
Msun = 1.99e33
pc = 3.086e18

def rk(t0, y0, h, func, *args):
    '''
    Gives one Runge-Kutta of fourth order step
    Usage:
    t1, y1 = rk(t0, y0, h, func, *args)
    '''
    k1 = func(y0, *args)
    k2 = func(y0 + h/2. * k1, *args)
    k3 = func(y0 + h/2. * k2, *args)
    k4 = func(y0 + h * k3, *args)
    y1 = y0 + h/6. * (k1 + 2.*k2 + 2.*k3 + k4)
    t1 = t0 + h
    return t1, y1

def dxdt(x, u):
    return u

def dudt(u, x):
    return -1. / x**2

# initial conditions
t = 0.
x = 1.
u = 0.

h = 1e-3
xmin = 1e-5
tlist = [t]
xlist = [x]
ulist = [u]
while x>xmin:
    t0, x0, u0 = t, x, u
    t, x = rk(t0, x0, h, dxdt, u0)
    t, u = rk(t0, u0, h, dudt, x0)
    tlist.append(t)
    xlist.append(x)
    ulist.append(u)

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(tlist, xlist, color='C0')
ax[0].set_ylabel(r'$r/r_0$', fontsize=16)
ax[1].plot(tlist, ulist, color='C1')
ax[1].set_xlabel(r'$\alpha t$', fontsize=16)
ax[1].set_ylabel(r'$v/(\alpha r_0)$', fontsize=16)
plt.text(0, -20, r'$\alpha=\sqrt{GM/r_0^3}$')
plt.savefig('homologous_collapse.png', dpi=100, bbox_inches='tight')
plt.close()

# compare tff
#tff = np.sqrt(3. * np.pi / 32 / G / rho0) = ... = np.pi / 2. / np.sqrt(2.) * r0**3 / G / M
#tauff = tff / alpha
tauff = np.pi / 2 / np.sqrt(2)
print('(Tnum-Tteo)/Tteo = {:.3f} %'.format(100. * (t-tauff)/tauff))

# compute typical values
# input parameters
M = Msun
r0 = 0.1 * pc
alpha = np.sqrt(G * M / r0**3)
Myr = 31556926.08 * 1e6
print('For M={:} Msun, r0={:} pc:'.format(M / Msun, r0 / pc))
print('tff = {:.2f} Myr'.format(t / alpha / Myr))

#!/usr/bin/env python3
'''
Solves Lane-Emden equation
d(-z**2 * dwdz)dz = z**2 * w**n
(Pols, Eq.4.7, p.47)

System of equations:
Defining
theta = -z**2 * dwdz,
we have
dwdz = -theta / z**2
dthetadz = z**2 * w**n

Boundaries:
w = 1 at z = 0
dwdz = 0 at z = 0
'''
import numpy as np
import matplotlib.pyplot as plt

def rk4(x0, y0, h, func, *args):
    '''
    Gives one Runge-Kutta of fourth order step
    Usage:
    t1, y1 = rk(t0, y0, h, func, *args)
    '''
    k1 = func(x0, y0, *args)
    k2 = func(x0 + h/2., y0 + h/2 * k1, *args)
    k3 = func(x0 + h/2., y0 + h/2 * k2, *args)
    k4 = func(x0 + h, y0 + h * k3, *args)
    y1 = y0 + h/6. * (k1 + 2.*k2 + 2.*k3 + k4)
    x1 = x0 + h
    return x1, y1

def dwdz(z, w, theta):
    if z == 0:
        return 0
    return -theta / z**2

def dthetadz(z, theta, w, n):
    return z**2 * w**n

###################
###################
# POLYTROPIC INDEX
n = 1
###################
###################

# initial conditions
z = 0.
w = 1.
theta = 0.

h = 1e-3
wmin = 1e-5
zl = [z]
wl = [w]
thetal = [theta]
while w > wmin:
    z0, w0, theta0 = z, w, theta
    z, w = rk4(z0, w0, h, dwdz, theta0)
    z, theta = rk4(z0, theta0, h, dthetadz, w0, n)
    zl.append(z)
    wl.append(w)
    thetal.append(theta)
zl = np.array(zl)

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# plot numerical solution
plt.figure()
plt.plot(zl, wl, lw=3, alpha=.5, label='numerical solution')
plt.xlabel('$z$', fontsize=16)
plt.ylabel('$w(z)$', fontsize=16)
plt.title(f'Polytropic index $n={n}$')

# plot analytical solution (when available)
if n == 0:
    ww = 1 - zl**2 / 6
    plt.plot(zl, ww, ':k', label='analytic solution')
elif n == 1:
    ww = np.ones(len(zl))
    ww[1:] = np.sin(zl[1:]) / zl[1:]
    plt.plot(zl, ww, ':k', label='analytic solution')
elif n == 5:
    ww = (1 + zl**2 / 3)**(-.5)
    plt.plot(zl, ww, ':k', label='analytic solution')
else:
    print('No analytical solution available!')
plt.legend()
plt.savefig(f'politrope_n{n}.png', dpi=100, bbox_inches='tight')
plt.close()

#!/usr/bin/env python3
'''
Plots related to the Bremsstrahlung emission

References:
Ghisellini p.26
Rybicki Eqs. 5.4, 5.10
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# constants
c = 3e10
k = 1.380658e-16
h = 6.6260755e-27

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# create figs/ folder
if not os.path.isdir('figs/'):
    out = os.system('mkdir figs')

##################
# Ghisellini p.26

# input
R = 1e15
Tarr = [1e4, 1e5, 1e6, 1e7]
narr = np.array([1e10, 1e12, 1e14, 1e16, 1e18])

# fixed
Z = 1.
g = 1.
numin, numax, nnu = 1e10, 1e20, 1000
nu = numin * (numax/numin)**np.linspace(0, 1, nnu)

# intensity n
Imin, Imax = 1e30, 1e-30
T = 1e7
B = 2. * h * nu**3 / c**2 / (np.exp(h*nu/(k*T)) - 1.)
plt.figure()
for n in narr:
    alpha = 3.7e8 * Z**2 * n**2 * T**(-.5) * nu**(-3) * (1. - np.exp(-h*nu / (k*T))) * g
    I = B * (1. - np.exp(-alpha * R))
    plt.plot(nu, I, label='n={:.0e} cm-3'.format(n))
    Imin = np.min([Imin, I[0]])
    Imax = np.max([Imax, I.max()])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu\,[\mathrm{Hz}]$', fontsize=16)
plt.ylabel(r'$I_{\nu}\,[\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}\,Hz^{-1}}]$', fontsize=16)
plt.ylim((Imin, Imax))
plt.legend()
plt.savefig('figs/density_dep.png', dpi=100, bbox_inches='tight')
plt.close()

# intensity T
Imin, Imax = 1e30, 1e-30
plt.figure()
n = 1e10
for T in Tarr:
    B = 2. * h * nu**3 / c**2 / (np.exp(h*nu/(k*T)) - 1.)
    alpha = 3.7e8 * Z**2 * n**2 * T**(-.5) * nu**(-3) * (1. - np.exp(-h*nu / (k*T))) * g
    I = B * (1. - np.exp(-alpha * R))
    plt.plot(nu, I, label='T={:.0e} K'.format(T))
    Imin = np.min([Imin, I[0]])
    Imax = np.max([Imax, I.max()])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu\,[\mathrm{Hz}]$', fontsize=16)
plt.ylabel(r'$I_{\nu}\,[\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}\,Hz^{-1}}]$', fontsize=16)
plt.ylim((Imin, Imax))
plt.legend()
plt.savefig('figs/temperature_dep.png', dpi=100, bbox_inches='tight')
plt.close()

###################
# Rybick Eqs. 5.4

x = np.linspace(-5, 5, 100)
v = np.zeros(len(x))
v[np.abs(x)-1.<0.] = 1.
plt.plot(x, v, lw=3)
plt.xlabel(r'$t/\tau$', fontsize=14)
plt.ylabel(r'$\dot{v}\,[\mathrm{arbitrary\,units}]$', fontsize=14)
ax = plt.gca()
ax.axes.yaxis.set_ticks([])
plt.savefig('figs/vdot.png', dpi=100, bbox_inches='tight')
plt.close()

###################
# Rybick Eqs. 5.10

x = np.linspace(.1, 2.5, 1000)
plt.plot(x, 1/x, lw=3, label=r'$\propto 1/v$')
plt.plot(x, 1/x**2, lw=3, label=r'$\propto 1/v^2$')
plt.xlabel(r'$v/[4Ze^2/\pi h]$', fontsize=14)
plt.ylabel(r'$b(v)\,[\mathrm{arbitrary\, units}]$', fontsize=14)
plt.yscale('log')
plt.legend()
plt.text(.2, 40, 'quantum limit', alpha=.6, fontsize=14)
plt.text(1.8, 1, 'classical limit', alpha=.6, fontsize=14)
plt.savefig('figs/bmin.png', dpi=100, bbox_inches='tight')
plt.close()

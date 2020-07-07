#!/usr/bin/env python3
'''
Relativistic charge emission
Rybicki & Lightman, Section 4.8
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14


# create figs/ folder
if not os.path.isdir('figs/'):
    out = os.system('mkdir figs')

# Eq. 4.98c
# (2gam / (1+gam^2 theta^2))^4
x = np.linspace(-2, 2, 100)
y = 1. / (1 + x**2)**4
plt.figure()
plt.plot(x, y, lw=3)
plt.xlabel(r'$\gamma\theta$', fontsize=16)
plt.ylabel(r'$(1 + \gamma^2\theta^2)^{-4}$')
plt.savefig('figs/beaming.png', dpi=100, bbox_inches='tight')
plt.close()

# Eqs. 3.18, 4.101, 4.103
bet = .95
gam2 = 1. / (1. - bet**2)

L = 1.
N = 128
x = np.linspace(-L, L, 128)
y = np.linspace(-L, L, 128)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2
sint2 = X**2 / R2
mu = Y / np.sqrt(R2)
rest = sint2 / R2

x = np.linspace(-L, L, 128)
y = np.linspace(-L/2, 1.5*L, 128)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2
sint2 = X**2 / R2
mu = Y / np.sqrt(R2)
para = sint2 / (1. - bet * mu)**6 / R2
perp = 1. / (1. - bet * mu)**4 * (1. - sint2 / gam2 / (1. - bet * mu)**2) / R2

plt.figure()
plt.contourf(np.log10(rest), levels=10, origin='lower', extent=[-L, L, -L, L])
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$y$', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'$\log_{10}[S_{\mathrm{rest}}/(q^2 a^2 / 4\pi c^3)]$', fontsize=16)
plt.savefig('figs/emission_rest.png', dpi=100, bbox_inches='tight')
plt.close()

plt.figure()
plt.contourf(np.log10(para), levels=10, origin='lower', extent=[-L, L, -L/2, 1.5*L])
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$y$', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'$\log_{10}[S_{\mathrm{parallel}}/(q^2 a^2 / 4\pi c^3)]$', fontsize=16)
plt.savefig('figs/emission_para.png', dpi=100, bbox_inches='tight')
plt.close()

plt.figure()
plt.contourf(np.log(perp), levels=10, origin='lower', extent=[-L, L, -L/2, 1.5*L])
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$y$', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'$\log_{10}[S_{\mathrm{perp}}/(q^2 a^2 / 4\pi c^3)]$', fontsize=16)
plt.savefig('figs/emission_perp.png', dpi=100, bbox_inches='tight')
plt.close()

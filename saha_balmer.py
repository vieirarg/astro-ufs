#!/usr/bin/env python3
'''
Boltzmann law + Saha equation
HI(n=1) + HI(n=2) + HII

References:
https://www.astro.princeton.edu/~gk/A403/ioniz.pdf
http://spiff.rit.edu/classes/phys440/lectures/saha/saha.html
http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F09/hydrogen-ionization.pdf
http://www.cambridge.org/us/files/5413/6681/8627/7706_Saha_equation.pdf
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# constants
pi = 3.141592653589793
e = 4.8032068e-10
h = 6.6260755e-27
k = 1.380658e-16
me = 9.1093897e-28

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# create figs/ folder
if not os.path.isdir('figs/'):
    out = os.system('mkdir figs')

E12 = 10.2 * 1.6e-12 # eV to erg
#E23 = 1.9 * 1.6e-12
Eion = 13.6 * 1.6e-12
T = np.linspace(1000, 20000, 5000)

g1 = 2 * 1**2
g2 = 2 * 2**2
g3 = 2 * 3**2

n = 1e14
alpha = g2/g1 * np.exp(-E12 / k / T)
#alpha = g3/g2 * np.exp(-E23 / k / T)
beta = (2. * pi * me * k * T / h**2)**1.5 * np.exp(-Eion / k / T)
ni = -beta/2. + np.sqrt((beta/2.)**2 + n * beta)
n1 = (n - ni) / (1. + alpha)
n2 = (n - ni) * alpha / (1. + alpha)

plt.figure()
plt.plot(T, n1 / n, lw=2, label=r'$n_1/n_{\mathrm{tot}}$')
plt.plot(T, 1e5 * n2 / n, lw=2, label=r'$n_2/n_{\mathrm{tot}}\times 10^{5}$')
plt.plot(T, ni / n, lw=2, label=r'$n_i/n_{\mathrm{tot}}$')
plt.legend()
plt.xlabel(r'$T\,[\mathrm{K}]$', fontsize=16)
plt.ylabel(r'$n/n_{\mathrm{tot}}$', fontsize=16)
plt.title(r'$\chi=13.6\,\mathrm{eV},\,n_{\mathrm{tot}}=n_1+n_2+n_i=10^{14}\,\mathrm{cm^{-3}}$')
plt.savefig('figs/saha_balmer.png', dpi=100, bbox_inches='tight')
plt.close()

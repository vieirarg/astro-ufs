#!/usr/bin/env python3
'''
Plots related to synchrotron emission

References:
Ghisellini, Cap. 4
Shu, Cap. 18
Rybicki, Cap. 6
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kv.html#scipy.special.kv
https://numpy.org/doc/stable/reference/routines.fft.html
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import kv

def integ(x, y):
    return np.trapz(y, x=x)

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# create figs/ folder
if not os.path.isdir('figs/'):
    out = os.system('mkdir figs')

# Modified Bessel function second kind
x = np.linspace(0.1, 10, 200)
plt.plot(x, kv(5./3., x), lw=3, label=r'$K_{5/3}$')
plt.plot(x, np.sqrt(np.pi/x) * np.exp(-x), lw=2, ls='--', label=r'$\sqrt{\pi/x}\,e^{-x}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$K_{5/3}(x)$', fontsize=16)
plt.savefig('figs/k53.png', dpi=100, bbox_inches='tight')
plt.close()

# Ghisellini, integral Eq. 4.15
x = np.linspace(0.1, 10, 200)
k53 = x * kv(5./3., x)
intF = np.array([integ(x[i:], k53[i:]) for i in np.arange(0, len(x)-1)])
plt.plot(x[:-1], x[:-1] * intF, lw=3)
plt.xlabel(r'$\nu/\nu_c$', fontsize=16)
plt.ylabel(r'$F(\nu/\nu_c)$', fontsize=16)
plt.savefig('figs/Fnu.png', dpi=100, bbox_inches='tight')
plt.close()

# spectrum
def peak(t, t0, sig):
    y = np.exp(-.5 * (t - t0)**2 / sig**2)
    #y[np.abs(t-t0)>3.*sig] = 0
    return y

tmin, tmax, dtb, dta = 0, 100, 10, .1
nsamp = 10
nt = np.int(nsamp * (tmax-tmin) / dta)
t = np.linspace(tmin, tmax, nt)
E = 0
npeak = np.int((tmax-tmin) / dtb)

# fake E signal
for i in range(npeak):
    t0 = (i + .5) * dtb
    E = E + peak(t, t0, dta/2.)

# signal
plt.figure()
plt.plot(t, E, color='C0')
plt.xlabel(r'$t\,[\mathrm{s}]$', fontsize=16)
plt.ylabel(r'$E(t)$', fontsize=16)
plt.title(r'$\Delta t_A={:}\,\mathrm{{s}},\,\Delta t_B={:}\,\mathrm{{s}}$'.format(dta, dtb))
plt.savefig('figs/Et.png', dpi=100, bbox_inches='tight')
plt.close()

# power spectrum
freq = np.fft.rfftfreq(nt, d=dta/nsamp)
ft = np.fft.rfft(E, norm='ortho')
ft2 = np.abs(ft)**2
plt.figure()
plt.plot(freq, ft2, color='C1')
plt.xlim(0, 1./dta)
plt.xlabel(r'$\nu\,[\mathrm{Hz}]$', fontsize=16)
plt.ylabel(r'$|\hat{E}(\omega)|^2$', fontsize=16)
plt.title(r'$\Delta \nu_A\sim{:}\,\mathrm{{Hz}},\,\Delta \nu_B={:}\,\mathrm{{Hz}}$'.format(1./dta, 1./dtb))
plt.savefig('figs/Ew2.png', dpi=100, bbox_inches='tight')
plt.close()

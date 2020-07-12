#!/usr/bin/env python3
'''
Rybicki & Lightman, prob. 4.13
https://en.wikipedia.org/wiki/Mollweide_projection
http://www.cosmo-ufes.org/uploads/1/3/7/0/13701821/cmb.dipole.plinio-2014.pdf
https://apod.nasa.gov/apod/ap140615.html
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product as prod

c = 29979245800.0
h = 6.6260755e-27
k = 1.380658e-16
T = 2.725

def bb(lbd, T, bet, theta):
    '''
    [lbd] = micron
    [theta] = deg
    '''
    Tb = Tboost(T, bet, theta)
    bb = 2. * h * c**3 / (lbd*1e-4)**5 / (np.exp(h * c / k / Tb / (lbd*1e-4)) - 1.)
    return bb

def Tboost(T, bet, theta):
    return T * np.sqrt(1. - bet**2) / (1. - bet * np.cos(theta))

def mollweide(l, b, R=1., l0=0., precision=1e-3):
    '''
    https://en.wikipedia.org/wiki/Mollweide_projection
    '''
    # Newton-Raphson
    t0 = 1e10
    t = b
    while np.abs(t-t0)>precision:
        t0 = t
        t = t - (2.*t + np.sin(2.*t) - np.pi*np.sin(b)) / (4. * np.cos(t)**2)
    # Mollweide transformation
    x = R * 2.*np.sqrt(2.)/np.pi * (l-l0) * np.cos(t)
    y = R * np.sqrt(2.) * np.sin(t)
    return x, y

# create figs/ folder
if not os.path.isdir('figs/'):
    out = os.system('mkdir figs')

#####################
# blackbody spectrum
#####################

# input parameters
lmin, lmax = 100., 10e3
Nl = 1000
lbd = lmin * (lmax / lmin)**np.linspace(0, 1, Nl)

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# plot blackbody
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))
bet = .1
ax[0].plot(lbd/1e3, bb(lbd, T, bet, 0.), label=r'$\theta=0^{\circ}$')
ax[0].plot(lbd/1e3, bb(lbd, T, bet, np.pi/2.), label=r'$\theta=90^{\circ}$')
ax[0].plot(lbd/1e3, bb(lbd, T, bet, np.pi), label=r'$\theta=180^{\circ}$')
ax[0].set_xscale('log')
ax[0].set_xlabel(r'$\lambda\,[\mathrm{mm}]$', fontsize=16)
ax[0].set_ylabel(r'$B_{\nu}(T, \theta)\,[\mathrm{erg\,s^{-1}\,ster^{-1}\,cm^{-1}}]$', fontsize=16)
ax[0].set_title(r'$\beta=0.1$')

v = 370e5 # 370 km/s -> cm/s
bet = v / c
ax[1].plot(lbd/1e3, bb(lbd, T, bet, 0.), label=r'$\theta=0^{\circ}$')
ax[1].plot(lbd/1e3, bb(lbd, T, bet, np.pi/2.), label=r'$\theta=90^{\circ}$')
ax[1].plot(lbd/1e3, bb(lbd, T, bet, np.pi), label=r'$\theta=180^{\circ}$')
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$\lambda\,[\mathrm{mm}]$', fontsize=16)
ax[1].set_ylabel(r'$B_{\nu}(T, \theta)\,[\mathrm{erg\,s^{-1}\,ster^{-1}\,cm^{-1}}]$', fontsize=16)
ax[1].set_title(r'$\beta=0.001$')
plt.legend()
plt.savefig('figs/cmb_bb.png', dpi=100, bbox_inches='tight')
plt.close()

##########
# cmb map
##########

# input parameters
T = 2.725
v = 370e5 # 370 km/s -> cm/s
bet = v / c
lp, bp = 263.99, 48.26 # polo
lp, bp = (lp-180.)/180. * np.pi, bp * np.pi/180.

# Mollweide projection
nres = 200
llist = np.linspace(-np.pi, np.pi, nres) # longitude
blist = np.linspace(-np.pi/2., np.pi/2., nres) # latitude

xlist = []
ylist = []
Tlist = []

for l, b in prod(llist, blist):
    x, y = mollweide(l, b)
    xlist.append(x)
    ylist.append(y)
    # theta = distance to the pole
    # http://www.krysstal.com/sphertrig.html
    cost = np.cos(np.pi/2.-b) * np.cos(np.pi/2.-bp) + \
           np.sin(np.pi/2.-b) * np.sin(np.pi/2.-bp) * np.cos(l-lp)
    theta = np.arccos(cost)
    Tlist.append(Tboost(T, bet, theta))

plt.figure(figsize=(10, 5))
delt = np.array(Tlist)/np.mean(Tlist)-1.
plt.scatter(xlist, ylist, c=delt, cmap=plt.cm.jet, alpha=.3)
cbar = plt.colorbar(label=r'$\Delta T/\langle T\rangle$')
xp, yp = mollweide(lp, bp)
plt.plot(xp, yp, 'w+', ms=12)
xp, yp = mollweide(lp-np.pi, -bp)
plt.plot(xp, yp, 'w+', ms=12)
plt.axis('off')

plt.savefig('figs/cmb_dipole.png', dpi=100, bbox_inches='tight')
plt.close()

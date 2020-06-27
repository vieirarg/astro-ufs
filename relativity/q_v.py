#!/usr/bin/env python3
'''
Electric field charge uniform movement
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product as prod

def electric_field(q, x, y, t, bet):
    '''
    Rybicki, Eqs. 4.70
    '''
    gam = 1. / np.sqrt(1. - bet**2)
    v = 3e10 * bet
    r3 = (gam**2 * (x - v * t)**2 + y**2)**1.5
    Ex = q * gam * (x - v * t) / r3
    Ey = q * gam * y / r3

    return Ex, Ey

# input
bet = .9
b = 1.
c = 3e10
q = 1.

# nicer plot font
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 14

# clean or create figs/ folder
if not os.path.isdir('figs/'):
    out = os.system('mkdir figs')

#######
# E(t)
#######
N = 200
gam = 1. / np.sqrt(1. - bet**2)
tmax = 3. * b / (bet*c) /gam
tarr = np.linspace(-tmax, tmax, N)
Ex, Ey = electric_field(q, 0., b, tarr, bet)
plt.figure()
plt.plot(tarr, Ex, label='$E_x(t)$')
plt.plot(tarr, Ey, label='$E_y(t)$')
plt.legend()
plt.xlabel(r'$t\,[\mathrm{s}]$', fontsize=16)
plt.ylabel(r'$\mathbf{E}(t)\,[\mathrm{statvolt\,cm^{-1}}]$', fontsize=16)
plt.grid()
plt.title(r'$q = {:.1f}\,\mathrm{{esu}}, \beta = {:.1f}, b = {:.1f}\,\mathrm{{cm}}$'.format(q, bet, b))
plt.savefig('figs/Et.png', dpi=100, bbox_inches='tight')
plt.close()

##########
# E(x, y)
##########

N = 64
L = .1
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
XX, YY = np.meshgrid(x, y)
Ex, Ey = np.zeros([N, N]), np.zeros([N, N])

#bet = .99
t = 0
xx = np.linspace(0.001, 1, 50)
for k, bet in enumerate(1. - np.exp(-5*xx)):
    for i, j in prod(range(N), range(N)):
        Ex[i, j], Ey[i, j] = electric_field(q, XX[i, j], YY[i, j], t, bet)

    # replace nan and zeroes for second smallest value
    # aesthetical purpose only
    Exmin = np.sort(list(set(Ex.flatten())))[1]
    Eymin = np.sort(list(set(Ey.flatten())))[1]
    Ex[np.isnan(Ex) + (Ex==0)] = Exmin
    Ey[np.isnan(Ey) + (Ey==0)] = Eymin

    img = np.hstack([Ey[:, ::-1], Ey])
    img = np.vstack([img[::-1, :], img])

    plt.figure(figsize=(7, 5))
    plt.contourf(np.log10(img), origin='lower', levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], \
                 cmap=plt.cm.viridis, extent=[-L, L, -L, L])
    plt.text(-.96*L, .88*L, r'$\beta={:.4f}$'.format(bet), fontsize=16, color='w')
    plt.xlabel('$x$'+' [cm]', fontsize=16)
    plt.ylabel('$y$'+' [cm]', fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label('$\log_{10}(E_y)\,[\mathrm{statvolt\,cm^{-1}}]$', fontsize=16)
    plt.savefig('figs/Ey_{:02d}.png'.format(k+1), dpi=100, bbox_inches='tight', pad_inches=.2)
    plt.close()

out = os.system('convert -delay 20 -loop 0 figs/Ey_*.png figs/Ey.gif')

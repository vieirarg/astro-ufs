#!/usr/bin/env python3
'''
Electric field charge uniform movement
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import product as prod
from scipy.special import k0, k1

def electric_field(q, x, y, t, bet):
    '''
    Rybicki, Eqs. 4.70, cgs units.

    Usage:
    Ex, Ey = electric_field(q, x, y, t, bet)
    '''
    gam = 1. / np.sqrt(1. - bet**2)
    v = 3e10 * bet
    r3 = (gam**2 * (x - v * t)**2 + y**2)**1.5
    Ex = q * gam * (x - v * t) / r3
    Ey = q * gam * y / r3
    return Ex, Ey

def integral(x, f):
    return np.trapz(f, x=x)

# constants
c = 3e10
me = 9.1093897e-28
hbar = 1.05457266e-27

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

# input
bet = .9
b = 1.
q = 1.

N = 200
gam = 1. / np.sqrt(1. - bet**2)
tmax = 3. * b / (bet*c) / gam
t = np.linspace(-tmax, tmax, N)
Ex, Ey = electric_field(q, 0., b, t, bet)
plt.figure()
plt.plot(t, Ex, label='$E_x(t)$')
plt.plot(t, Ey, label='$E_y(t)$')
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

# exponential spacing, to better sample larger beta
xx = np.linspace(0.001, 1, 50)
bet_arr = 1. - np.exp(-5*xx)
t = 0
for k, bet in enumerate(bet_arr):
    for i, j in prod(range(N), range(N)):
        Ex[i, j], Ey[i, j] = electric_field(q, XX[i, j], YY[i, j], t, bet)

    # replace nan and zeroes for second smallest value
    # aesthetical purpose only
    Exmin = np.sort(list(set(Ex.flatten())))[1]
    Eymin = np.sort(list(set(Ey.flatten())))[1]
    Ex[np.isnan(Ex) + (Ex==0)] = Exmin
    Ey[np.isnan(Ey) + (Ey==0)] = Eymin

    # Ex image
    img = np.hstack([Ex[:, ::-1], Ex])
    img = np.vstack([img[::-1, :], img])

    plt.figure(figsize=(7, 5))
    plt.contourf(np.log10(img), origin='lower', levels=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], \
                 cmap=plt.cm.viridis, extent=[-L, L, -L, L])
    plt.text(-.96*L, .88*L, r'$\beta={:.4f}$'.format(bet), fontsize=16, color='w')
    plt.xlabel('$x$'+' [cm]', fontsize=16)
    plt.ylabel('$y$'+' [cm]', fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label('$\log_{10}(E_x)\,[\mathrm{statvolt\,cm^{-1}}]$', fontsize=16)
    plt.savefig('figs/Ex_{:02d}.png'.format(k+1), dpi=100, bbox_inches='tight', pad_inches=.2)
    plt.close()

    # Ey image
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

out = os.system('convert -delay 20 -loop 0 figs/Ex_*.png figs/Ex.gif && rm figs/Ex_*.png')
out = os.system('convert -delay 20 -loop 0 figs/Ey_*.png figs/Ey.gif && rm figs/Ey_*.png')


###########
# SPECTRUM
###########

# input
q = 1.
b = 1.
bet = .9

v = c * bet
gam = 1. / np.sqrt(1. - bet**2)
N = 500
tmax = 20. * b / (bet*c) / gam
t = np.linspace(-tmax, tmax, N)
wmax = 50. / tmax
w = np.linspace(0, wmax, N+1)[1:]

#######
# E(w)
#######

Ex, Ey = electric_field(q, 0., b, t, bet)

# only coswt contributes, since Ey is even
# Rybicki, Eq. 4.71
Ew_brute_force = 1. / (2.*np.pi) * np.array([integral(t, Ey*np.cos(ww*t)) for ww in w])

# Rybicki, Eq. 4.72a
x1 = b * w / gam / v
Ew = q / (np.pi * b * v) * x1 * k1(x1)

plt.figure()
plt.plot(w, 100*(Ew-Ew_brute_force)/Ew)
plt.xlabel(r'$\omega\,[\mathrm{s^{-1}}]$', fontsize=16)
plt.ylabel(r'$\delta E_{\omega}/E_{\omega}\,[\mathrm{\%}]$', fontsize=16)
plt.xscale('log')
plt.title('Rybicki Eqs. 4.71 vs 4.72a')
plt.savefig('figs/Ew_residuals.png', dpi=100, bbox_inches='tight')
plt.close()

# Ew(b)
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\omega\,[\mathrm{s^{-1}}]$', fontsize=16)
plt.ylabel(r'$dW/dAd\omega\,[\mathrm{erg\,cm^{-2}\,s}]$', fontsize=16)
plt.title('Rybicki Eqs. 4.72b')
for b in [100, 200, 300, 400, 500]:
    x1 = b * w / gam / v
    dWdAdw = c * (q / (np.pi * b * v) * x1 * k1(x1))
    plt.plot(w, dWdAdw, label='b = {:} m'.format(b/100.))
plt.legend()
plt.savefig('figs/dWdw_b.png', dpi=100, bbox_inches='tight')
plt.close()

# Ew(b) zoom
plt.figure()
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\omega\,[\mathrm{s^{-1}}]$', fontsize=16)
plt.ylabel(r'$dW/dAd\omega\,[\mathrm{erg\,cm^{-2}\,s}]$', fontsize=16)
plt.title('Rybicki Eqs. 4.72b')
keep = w < 1e9
for b in [100, 200, 300, 400, 500]:
    x1 = b * w[keep] / gam / v
    dWdAdw = c * (q / (np.pi * b * v) * x1 * k1(x1))
    plt.plot(w[keep], dWdAdw, label='b = {:} m'.format(b/100.))
plt.legend()
plt.savefig('figs/dWdw_b_zoom.png', dpi=100, bbox_inches='tight')
plt.close()

# total spectrum, Eq. 4.74b
bmin = hbar / me / c
wmin = 1e8
wmax = 2. * gam * v / bmin
w = wmin * (wmax/wmin)**np.linspace(0, 1, 200)
x = w * bmin / gam / v
dWdw = 2. * q**2 * c / np.pi / v**2 * \
       (x * k0(x) * k1(x) - .5 * x**2 * (k1(x)**2 - k0(x)**2))

plt.figure()
plt.plot(w, dWdw)
plt.xscale('log')
plt.yscale('log')
yl = plt.ylim()
plt.plot([wmax, wmax], yl, 'k--')
plt.ylim(yl)
plt.xlabel(r'$\omega\,[\mathrm{s^{-1}}]$', fontsize=16)
plt.ylabel(r'$dW/d\omega\,[\mathrm{erg\,s}]$', fontsize=16)
plt.title('Rybicki Eqs. 4.74b')
plt.text(5e17, 1e-11, r'$\omega=\gamma v / b_{\mathrm{min}}$',\
         fontsize=16)
plt.savefig('figs/dWdw.png', dpi=100, bbox_inches='tight')
plt.close()

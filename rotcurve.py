#!/usr/bin/env python3
'''
Plots rotation curve of the Galaxy
Data retrieved from https://www.astro.uni-bonn.de/hisurvey/profile/index.php
l=20, 25, 30, ..., 80; b=0
Downloaded files renamed as 'spectrum##.txt', where ##=gal. longitude
'''
import numpy as np
import matplotlib.pyplot as plt

def line_read(l, dir0='linha21cm/'):
    fname = dir0 + 'spectrum{:.0f}.txt'.format(l)
    f = open(fname, 'r')
    r = f.readlines()
    f.close()
    i = np.where(np.array([rr.find('%%')==0 for rr in r]))[0][-1]
    v = np.array([rr.split()[0] for rr in r[i+1:-2]], dtype=np.float)
    flux = np.array([rr.split()[1] for rr in r[i+1:-2]], dtype=np.float)
    return v, flux

R0, V0 = 8., 220.

dir0 = 'linha21cm/' # folder containing data files
R = []
V = []
df = 2. # flux threshold for vmax detection
plt.figure()
for l in np.arange(20, 85, 5):
    v, flux = line_read(l, dir0=dir0)
    j = np.abs(flux[v>0.]-df) == np.min(np.abs(flux[v>0.]-df))
    vmax = v[v>0][j]
    R.append(R0 * np.sin(l * np.pi/180.))
    V.append(vmax + V0 * np.sin(l * np.pi/180.))
    plt.plot(v, flux + 100 * (l-20)/5.)
    plt.text(150., flux[-1] + 100 * (l-20)/5., f'$l={l}^{{\circ}}$')
plt.xlabel('v [km/s]')
plt.ylabel('flux')
plt.savefig('spectra.png', dpi=100, bbox_inches='tight')
plt.close()

plt.figure()
plt.scatter(R, V)
plt.xlabel('R [kpc]')
plt.ylabel('V [km/s]')
plt.xlim(-.5, 17.5)
plt.ylim(150, 325)
plt.savefig('rotcurve.png', dpi=100, bbox_inches='tight')
plt.close()

for i in range(len(R)):
    print(f'{R[i]:.1f} & {V[i][0]:.0f} \\\\')

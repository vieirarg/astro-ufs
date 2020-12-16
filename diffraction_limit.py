#!/usr/bin/env python3
'''
https://docs.astropy.org/en/stable/api/astropy.convolution.AiryDisk2DKernel.html
https://docs.astropy.org/en/stable/convolution/kernels.html?highlight=convolution%20image
'''
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import AiryDisk2DKernel
from astropy.convolution import convolve

Rstar = 28 # mas
lbd = 2550. # UV
D = 2.4 # Hubble

radius = 1.22 * (lbd * 1e-10) / D * (180/np.pi * 60 * 60 * 1000) # mas
airy = AiryDisk2DKernel(radius)
airy_len = airy.shape[0]
airy_prof = airy.array[airy_len // 2, :]
x = np.linspace(-4 * radius, 4 * radius, airy_len) # default size is 8 8 radius

cmap = plt.cm.hot

# kernel
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(np.sqrt(airy.array), interpolation='none', origin='lower', cmap=cmap, \
             extent=[-4 * radius, 4 * radius, -4 * radius, 4 * radius])
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
#plt.colorbar()
plt.title('Diffraction pattern')
ax[1].plot(x, airy.array[airy.shape[0]//2, :])
plt.show()

# perfectly resolved image
# nicer resolution
nres = 512
size = 100.
x = np.linspace(-size/2, size/2, nres)
y = np.linspace(-size/2, size/2, nres)
X, Y = np.meshgrid(x, y)
img = np.zeros([nres, nres])
img[X**2 + Y**2 <= Rstar**2] = 1.
plt.figure()
plt.imshow(img, origin='lower', cmap=cmap, \
           extent=[-size/2, size/2, -size/2, size/2])
plt.xlabel('x [mas]')
plt.ylabel('y [mas]')
plt.colorbar()
plt.title('Original image')
plt.show()

# convolution
# smaller res, so it does not take too long to compute
nres = 128
size = 100.
x = np.linspace(-size/2, size/2, nres)
y = np.linspace(-size/2, size/2, nres)
X, Y = np.meshgrid(x, y)
img = np.zeros([nres, nres])
img[X**2 + Y**2 <= Rstar**2] = 1.
obs = convolve(img, airy)
plt.figure()
plt.imshow(obs, origin='lower', cmap=cmap, \
           extent=[-size/2, size/2, -size/2, size/2])
plt.xlabel('x [mas]')
plt.ylabel('y [mas]')
#plt.colorbar()
plt.title('Convolution')
plt.show()

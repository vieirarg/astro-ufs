#!/usr/bin/env python3
'''
'''
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# symbols
gam, bet, Ex, Ey, Ez, Bx, By, Bz = sym.symbols('gamma, beta, E_x, E_y, E_z, B_x, B_y, B_z')
e, c, m0 = sym.symbols('e, c, m0')
gamu, ux, uy, uz = sym.symbols('gamma_u, u_x, u_y, u_z')

# definitions
Fcov = sym.Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, Bz, -By], [Ey, -Bz, 0, Bx], [Ez, By, -Bx, 0]])
L = sym.Matrix([[gam, -bet*gam, 0, 0], [-bet*gam, gam, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
G = sym.Matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
Ltil = G * L * G.transpose()
U = sym.Matrix([gamu * c, gamu * ux, gamu * uy, gamu * uz])
P = m0 * U
Fmix = G * Fcov

# Eq. 4.84
sym.pprint(sym.simplify(e/c * Fmix * U))

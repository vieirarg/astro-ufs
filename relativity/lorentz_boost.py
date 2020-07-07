#!/usr/bin/env python3
'''
Matricial algebra
Lorentz boost
'''
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# symbols
gam, bet, Ex, Ey, Ez, Bx, By, Bz = sym.symbols('gamma, beta, E_x, E_y, E_z, B_x, B_y, B_z')

# definitions
Fcont = sym.Matrix([[0, Ex, Ey, Ez], [-Ex, 0, Bz, -By], [-Ey, -Bz, 0, Bx], [-Ez, By, -Bx, 0]])
Fcov = sym.Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, Bz, -By], [Ey, -Bz, 0, Bx], [Ez, By, -Bx, 0]])
L = sym.Matrix([[gam, -bet*gam, 0, 0], [-bet*gam, gam, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
Ltil = sym.Matrix([[gam, bet*gam, 0, 0], [bet*gam, gam, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
G = sym.Matrix([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# contravariant boost
F1 = L.transpose() * Fcont * L
F1.simplify()

F2 = L * Fcont * L.transpose()
F2.simplify()

print('contravariant')

print(r'\begin{equation}')
sym.print_latex(F1)
print(r'\end{equation}')

print(r'\begin{equation}')
sym.print_latex(F2)
print(r'\end{equation}')

# covariant boost
F1 = Ltil.transpose() * Fcov * Ltil
F1.simplify()

F2 = Ltil * Fcov * Ltil.transpose()
F2.simplify()

print('contravariant')

print(r'\begin{equation}')
sym.print_latex(F1)
print(r'\end{equation}')

print(r'\begin{equation}')
sym.print_latex(F2)
print(r'\end{equation}')

# invariant contraction
# F_mn * F^mn
# A:B := Trace(A*transpose(B)) = A(ij)B(ij)
Fcont = sym.Matrix([[0, -Ex, -Ey, -Ez], [Ex, 0, -Bz, By], [Ey, Bz, 0, -Bx], [Ez, -By, Bx, 0]])
Fcov = sym.Matrix([[0, Ex, Ey, Ez], [-Ex, 0, -Bz, By], [-Ey, Bz, 0, -Bx], [-Ez, -By, Bx, 0]])
sym.pprint(np.trace(Fcov*Fcont.transpose()))

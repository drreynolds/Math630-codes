#!/usr/bin/env python3
#
# Script to use francis1 and francis2 functions to compute Schur
# decompositions of some matrices.
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import numpy as np
from numpy.linalg import norm
import warnings
from upper_hess import upper_hess
from francis1 import francis1
from francis_step import francis_step

# suppress runtime warnings arising from built-in matrix-matrix products
warnings.filterwarnings("ignore", category=RuntimeWarning)

# general solver parameters
maxit = 100
tol = 1e-6

# symmetric matrix
n = 10
A = np.random.rand(n,n)
A = A + A.T
print("first matrix A (symmetric):")
print("  ||A - A^T|| = ",norm(A-A.T),"\n")

# convert to upper-Hessenberg form
H, Q = upper_hess(A)
print("upper-Hessenberg H = Q^T A Q:")
print("checks:")
print("  norm of lower portion of H = ", norm(H - np.triu(H,-1)))
print("  ||Q^T Q - I|| = ", norm(Q.T @ Q - np.eye(n)))
print("  ||Q Q^T - I|| = ", norm(Q @ Q.T - np.eye(n)))
print("  ||Q^T A Q - H|| = ", norm(Q.T @ A @ Q - H))
print("  norm of subdiagonal of H = ", norm(np.diag(H,-1)),"\n")
input("Press Enter to continue...")


# run Francis1 with Rayleigh quotient shift
print("Running francis1 with Rayleigh quotient shift:")
T, U, its = francis1(H,maxit,tol,0,1)
print("checks:")
print("  ||U^T U - I|| = ", norm(U.T @ U - np.eye(n)))
print("  ||U U^T - I|| = ", norm(U @ U.T - np.eye(n)))
print("  ||U^T H U - T|| = ", norm(U.T @ H @ U - T))
print("  ||U^T Q^T A Q U - T|| = ", norm(U.T @ Q.T @ A @ Q @ U - T))
print("  norm of lower triangle of T = ", norm(np.tril(T,-1)), "\n")
input("Press Enter to continue...")

# run Francis1 with Wilkinson shift
print("Running francis1 with Wilkinson shift:")
T, U, its = francis1(H,maxit,tol,1,1)
print("checks:")
print("  ||U^T U - I|| = ", norm(U.T @ U - np.eye(n)))
print("  ||U U^T - I|| = ", norm(U @ U.T - np.eye(n)))
print("  ||U^T H U - T|| = ", norm(U.T @ H @ U - T))
print("  ||U^T Q^T A Q U - T|| = ", norm(U.T @ Q.T @ A @ Q @ U - T))
print("  norm of lower triangle of T = ", norm(np.tril(T,-1)),"\n")
input("Press Enter to continue...")



# non-symmetric matrix
n = 10     # must be even
v = np.random.rand(n,n//2) + 1j*np.random.rand(n,n//2)
V = np.hstack([v, np.conj(v)])
d = np.random.rand(n//2) + 1j*np.random.rand(n//2)
D = np.diag(np.hstack( [d, np.conj(d)] ))
A = np.real(V @ D @ np.linalg.inv(V))
print("\nsecond matrix A (non-symmetric):")
print("  ||A - A^T|| = ", norm(A-A.T), "\n")

# convert to upper-Hessenberg form
H, Q = upper_hess(A)
print("upper-Hessenberg H = Q^T A Q:")
print("checks:")
print("  norm of lower portion of H = ", norm(H - np.triu(H,-1)))
print("  ||Q^H Q - I|| = ", norm(Q.T.conj() @ Q - np.eye(n)))
print("  ||Q Q^H - I|| = ", norm(Q @ Q.T.conj() - np.eye(n)))
print("  ||Q^H A Q - H|| = ", norm(Q.T.conj() @ A @ Q - H))
print("  norm of subdiagonal of H = ", norm(np.diag(H,-1)), "\n")
input("Press Enter to continue...")


# run Francis1 with Rayleigh quotient shift
print("Running francis1 with Rayleigh quotient shift:")
T, U, its = francis1(H,maxit,tol,0,1)
print("checks:")
print("  ||U^H U - I|| = ", norm(U.T.conj() @ U - np.eye(n)))
print("  ||U U^H - I|| = ", norm(U @ U.T.conj() - np.eye(n)))
print("  ||U^H H U - T|| = ", norm(U.T.conj() @ H @ U - T))
print("  ||U^H Q^H A Q U - T|| = ", norm(U.T.conj() @ Q.T.conj() @ A @ Q @ U - T))
print("  norm of lower triangle of T = ", norm(np.tril(T,-1)), "\n")
input("Press Enter to continue...")

# run Francis1 with Wilkinson shift
print("Running francis1 with Wilkinson shift:")
T, U, its = francis1(H,maxit,tol,1,1)
print("checks:")
print("  ||U^H U - I|| = ", norm(U.T.conj() @ U - np.eye(n)))
print("  ||U U^H - I|| = ", norm(U @ U.T.conj() - np.eye(n)))
print("  ||U^H H U - T|| = ", norm(U.T.conj() @ H @ U - T))
print("  ||U^H Q^H A Q U - T|| = ", norm(U.T.conj() @ Q.T.conj() @ A @ Q @ U - T))
print("  norm of lower triangle of T = ", norm(np.tril(T,-1)), "\n")

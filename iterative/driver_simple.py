#!/usr/bin/env python3
#
# Script to test simple iterative methods on our 2D Laplace problem
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import time
import numpy as np
import stationary
from numpy.linalg import norm
from laplace_2D import laplace_2D
from scipy.sparse.linalg import spsolve


##################
# utility routines

def f(x,y):
    """forcing function for our test problem"""
    return 0.0

def g(x,y):
    """boundary condition function for our test problem"""
    tol = np.sqrt(np.finfo(float).eps)
    if (abs(x-0.0) < tol):
        return 0.0
    if (abs(x-1.0) < tol):
        return y
    if (abs(y-0.0) < tol):
        return (x-1.0)*np.sin(x)
    if (abs(y-1.0) < tol):
        return x*(2.0-x)

def initial_guess(M):
    """initial guess for our iterative methods, M is the number of intervals"""
    return np.zeros((M-1)*(M-1), dtype=float)

##################
# script

if __name__ == "__main__":

    # set grid sizes to try
    Mvals = [20,40,80]

    # set stopping tolerance
    tol = 1.0e-6

    # set maximum allowed number of iterations
    maxiters = 10000

    # set SOR relaxation parameter values to try
    omega_vals = [0.8, 1.0, 1.5, 1.8, 1.9, 1.95, 1.97, 2.0]

    # loop over problem sizes, running each iterative solver
    for M in Mvals:

        print("Problem size =", M)
        A,b = laplace_2D(M,f,g)
        xtrue = spsolve(A, b)

        # run Jacobi method
        x0 = initial_guess(M)
        tic = time.perf_counter()
        for niters in range(maxiters):
            x = x0.copy()
            x = stationary.jacobi(x, b)
            if (norm(x - x0)/norm(x0+1e-12) < tol):
                break
            x0 = x.copy()
        toc = time.perf_counter()
        print("  Jacobi:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters+1, toc-tic, norm(x - xtrue), norm(b - A@x)))

        # run [forward] Gauss-Seidel method
        x0 = initial_guess(M)
        tic = time.perf_counter()
        for niters in range(maxiters):
            x = x0.copy()
            x = stationary.fwd_gauss_seidel(x, b)
            if (norm(x - x0)/norm(x0+1e-12) < tol):
                break
            x0 = x.copy()
        toc = time.perf_counter()
        print("  GS:        niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters+1, toc-tic, norm(x - xtrue)/norm(xtrue), 
                 norm(b - A@x)/norm(b)))

        # run Red-Black Gauss-Seidel method
        x0 = initial_guess(M)
        tic = time.perf_counter()
        for niters in range(maxiters):
            x = x0.copy()
            x = stationary.red_black_gauss_seidel(x, b)
            if (norm(x - x0)/norm(x0+1e-12) < tol):
                break
            x0 = x.copy()
        toc = time.perf_counter()
        print("  RB-GS:     niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters+1, toc-tic, norm(x - xtrue)/norm(xtrue), 
                 norm(b - A@x)/norm(b)))

        # run Symmetric Gauss-Seidel method
        x0 = initial_guess(M)
        tic = time.perf_counter()
        for niters in range(maxiters):
            x = x0.copy()
            x = stationary.symmetric_gauss_seidel(x, b)
            if (norm(x - x0)/norm(x0+1e-12) < tol):
                break
            x0 = x.copy()
        toc = time.perf_counter()
        print("  Sym-GS:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (2*(niters+1), toc-tic, norm(x - xtrue)/norm(xtrue), 
                 norm(b - A@x)/norm(b)))

        # run SOR method for each omega value
        for omega in omega_vals:
            x0 = initial_guess(M)
            tic = time.perf_counter()
            for niters in range(maxiters):
                x = x0.copy()
                x = stationary.sor(x, b, omega=omega)
                if (norm(x - x0)/norm(x0+1e-12) < tol):
                    break
                x0 = x.copy()
            toc = time.perf_counter()
            print("  SOR(%.2f): niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
                  % (omega, niters+1, toc-tic, norm(x - xtrue)/norm(xtrue), 
                     norm(b - A@x)/norm(b)))


        
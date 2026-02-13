#!/usr/bin/env python3
#
# Script to test descent methods on our 2D Laplace problem
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import time
import numpy as np
import stationary
from steepest_descent import steepest_descent
from conjugate_gradient import conjugate_gradient
from numpy.linalg import norm
from laplace_2D import laplace_2D
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.sparse import tril
from scipy.sparse import triu
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve_triangular as trisolve


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
    Mvals = [20,40,80,160,320]

    # set stopping tolerance
    tol = 1.0e-6

    # set maximum allowed number of iterations
    maxiters = 10000

    # loop over problem sizes, running each iterative solver
    for M in Mvals:

        print("Problem size =", M)
        A,b = laplace_2D(M,f,g)
        xtrue = spsolve(A, b)

        # create preconditioners
        D = diags(A.diagonal(), format='csc')
        E = -tril(A,-1)
        F = -triu(A,1)
        # Jacobi: M = D, so P = D^{-1}
        PJacobi = lambda x: trisolve(D, x, lower=True)
        # symmetric GS: M = L D^{-1} U, where L=D-E, U=D-F
        #   so P = U^{-1} D L^{-1}
        LSGS = csc_matrix(D-E)
        USGS = csc_matrix(D-F)
        PSGS = lambda x: trisolve(USGS, D@(trisolve(LSGS, x, lower=True)), lower=False)
        # SSOR: M = om/(2-om)*L*D^{-1}*U, where L=1/om*D-E, U=1/om*D-F
        #   so P = (2-om)/om*U^{-1}*D*L^{-1}
        om = 1.5
        LSOR1 = csc_matrix((1.0/om)*D-E)
        USOR1 = csc_matrix((1.0/om)*D-F)
        PSOR1 = lambda x: ((2.0-om)/om)*trisolve(USOR1, D@(trisolve(LSOR1, x, lower=True)), lower=False)
        om = 1.9
        LSOR2 = csc_matrix((1.0/om)*D-E)
        USOR2 = csc_matrix((1.0/om)*D-F)
        PSOR2 = lambda x: ((2.0-om)/om)*trisolve(USOR2, D@(trisolve(LSOR2, x, lower=True)), lower=False)
    
        # run un-preconditioned steepest descent
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = steepest_descent(A, None, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  SD:        niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue), norm(b - A@x)))

        # run steepest descent with Jacobi preconditioning
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = steepest_descent(A, PJacobi, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  SD-Jac:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run steepest descent with symmetric Gauss-Seidel preconditioning
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = steepest_descent(A, PSGS, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  SD-SGS:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run steepest descent with SOR preconditioner 1
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = steepest_descent(A, PSOR1, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  SD-SOR1:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run steepest descent with SOR preconditioner 2
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = steepest_descent(A, PSOR2, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  SD-SOR2:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              %(niters,toc-tic,norm(x-xtrue)/norm(xtrue),norm(b-A@x)/norm(b)))
            
        # run un-preconditioned conjugate gradient
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = conjugate_gradient(A, None, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  CG:        niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue), norm(b - A@x)))

        # run conjugate gradient with Jacobi preconditioning
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = conjugate_gradient(A, PJacobi, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  CG-Jac:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run conjugate gradient with symmetric Gauss-Seidel preconditioning
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = conjugate_gradient(A, PSGS, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  CG-SGS:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run conjugate gradient with SOR preconditioner 1
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = conjugate_gradient(A, PSOR1, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  CG-SOR1:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run conjugate gradient with SOR preconditioner 2
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, niters, _ = conjugate_gradient(A, PSOR2, x0, b, maxiters, tol)
        toc = time.perf_counter()
        print("  CG-SOR2:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))
            

        
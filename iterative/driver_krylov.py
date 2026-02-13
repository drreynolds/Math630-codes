#!/usr/bin/env python3
#
# Script to test built-in Krylov methods on our 2D Laplace problem
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import time
import numpy as np
from numpy.linalg import norm
from laplace_2D import laplace_2D
import scipy.sparse.linalg as la
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
        xtrue = la.spsolve(A, b)

        # create preconditioner
        D = diags(A.diagonal(), format='csc')
        E = -tril(A,-1)
        F = -triu(A,1)
        # SSOR: M = om/(2-om)*L*D^{-1}*U, where L=1/om*D-E, U=1/om*D-F
        #   so P = (2-om)/om*U^{-1}*D*L^{-1}
        om = 1.5
        L = csc_matrix((1.0/om)*D-E)
        U = csc_matrix((1.0/om)*D-F)
        SOR = lambda x: ((2.0-om)/om)*trisolve(U, D@(trisolve(L, x, lower=True)), lower=False)
        P = la.LinearOperator(matvec=SOR, rmatvec=SOR, shape=A.shape, dtype=float)

        # custom iteration counter function since that isn't generally returned
        niters=0
        def count_iters(xk):
            global niters
            niters += 1

        # run pcg
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.cg(A, b, x0, rtol=tol, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  PCG:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run minres
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.minres(A, b, x0, rtol=tol, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  MINRES:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run gmres with restart size of 10
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.gmres(A, b, x0, rtol=tol, restart=10, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  GMRES(10): niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run gmres with restart size of 20
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.gmres(A, b, x0, rtol=tol, restart=20, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  GMRES(20): niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run bicg
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.bicg(A, b, x0, rtol=tol, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  BICG:      niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run cgs
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.cgs(A, b, x0, rtol=tol, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  CGS:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run bicgstab
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.bicgstab(A, b, x0, rtol=tol, maxiter=maxiters, M=P, callback=count_iters)
        toc = time.perf_counter()
        print("  BICGSTAB:  niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

        # run qmr
        niters=0
        x0 = initial_guess(M)
        tic = time.perf_counter()
        x, _ = la.qmr(A, b, x0, rtol=tol, maxiter=maxiters, M1=P, M2=P,callback=count_iters)
        toc = time.perf_counter()
        print("  QMR:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e" 
              % (niters, toc-tic, norm(x - xtrue)/norm(xtrue), norm(b - A@x)/norm(b)))

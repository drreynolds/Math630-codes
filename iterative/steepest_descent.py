#!/usr/bin/env python3

def steepest_descent(A, Pinv, x, b, maxit, tol):
    """
    Utility to perform preconditioned steepest descent.

    Inputs:
        A is a sparse matrix (n x n)
        Pinv is a function for the preconditioner solve, 
             i.e. to y = Pinv(z) computes y = P^{-1}@z 
             To run without a preconditioner, supply "None"
        x is the initial guess vector (n)
        b is the right-hand side vector (n)
        maxit is the maximum number of allowed iterations
        tol is the requested relative solution tolerance

    Outputs: x, iters, errnorm
        x is the final iterate vector (n)
        iters is the number of iterations performed
        errnorm is ||xnew-x||/||x||

    Daniel R. Reynolds
    Math 630 @ UMBC
    Spring 2026
    """
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm

    # if Pinv==None, supply a dummy
    if (Pinv is None):
        Pinv = lambda z: z

    # allocate vector storage
    r = np.zeros_like(x)
    p = np.zeros_like(x)
    q = np.zeros_like(x)

    # perform algorithm
    r = b - A@x
    p = Pinv(r)
    for iter in range(1,maxit+1):
        q = A@p
        alpha = dot(p,r) / dot(p,q)
        x = x + alpha*p
        r = r - alpha*q
        dxnorm = abs(alpha)*norm(p)
        xnorm = norm(x)
        p = Pinv(r)
        if (dxnorm <= xnorm*tol):
            break
    return x, iter, dxnorm/xnorm

#!/usr/bin/env python3
#
# Utility function to create a sparse matrix and right-hand side corresponding 
# to the linear system
# $$-\Delta u(x,y) = f(x,y), \quad (x,y) \in [0,1]^2$$
# with boundary conditions
# $$u(x,y) = g(x,y), \quad (x,y) \in \partial [0,1]^2$$
# where $u \in \Real$ is discretized using Nx points in the x-direction and 
# Ny points in the y-direction.  The Laplace operator is discretized using 
# the standard 2nd-order 5 point stencil, and the boundary condition 
# function is passed in as an input.
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

##################
# utility routines

def laplace_2D(M, f, g):
    r"""
    Usage: A,b = laplace_2D(M, f, g)

    inputs:
        M    # intervals in each of the x and y directions
        f    # forcing function
        g    # boundary condition function

    outputs:
        A    REAL ((M-1)*(M-1)) x ((M-1)*(M-1)) sparse (CSC) matrix
        b    REAL ((M-1)*(M-1)) vector
    """

    # set number of interior finite difference points in each direction
    N = M - 1

    # set indexing function from 2D physical space to 1D index space
    def ij(i,j):
        return(j*N + i)

    # initialize the data and row/column index arrays for A
    nrows = N*N
    nnzmax = nrows*5
    rows = np.zeros(nnzmax, dtype=float)
    cols = np.zeros(nnzmax, dtype=float)
    vals = np.zeros(nnzmax, dtype=float)

    # initialize the RHS vector b
    b = np.zeros(nrows, dtype=float)

    # set differencing constants
    h = 1.0/M
    h2 = h*h

    # iterate over the domain
    idx = 0
    for iy in range(N):
        for ix in range(N):

            # set the x,y location and neighboring points
            x = (ix+1)*h
            y = (iy+1)*h

            # set the forcing term into the RHS
            b[ij(ix,iy)] = f(x, y)*h2

            # set the matrix entries for this row of D
            #   diagonal
            rows[idx] = ij(ix,iy)
            cols[idx] = ij(ix,iy)
            vals[idx] = 4.0
            idx += 1

            #   x-left
            if (ix > 0):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix-1,iy)
                vals[idx] = -1.0
                idx += 1
            else:
                b[ij(ix,iy)] += g(0.0,y)

            #   x-right
            if (ix < N-1):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix+1,iy)
                vals[idx] = -1.0
                idx += 1
            else:
                b[ij(ix,iy)] += g(1.0,y)

            #   y-left
            if (iy > 0):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix,iy-1)
                vals[idx] = -1.0
                idx += 1
            else:
                b[ij(ix,iy)] += g(x,0.0)

            #   y-right
            if (iy < N-1):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix,iy+1)
                vals[idx] = -1.0
                idx += 1
            else:
                b[ij(ix,iy)] += g(x,1.0)

    A = csc_matrix(coo_matrix((vals, (rows,cols)), shape=(nrows,nrows)))
    return A, b

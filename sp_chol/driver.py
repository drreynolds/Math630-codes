#!/usr/bin/env python3
#
# Script to demonstrate sparse Cholesky factorizations (and reorderings) -- this requires the scikit-sparse pakage.
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from sksparse.cholmod import cholesky
import matplotlib.pyplot as plt


##################
# utility routines

def makeplots(D):
    """
    Usage: makeplots(D)

    This routine creates 6 plots:
       1. D and its R factor
       2. D using the approximate minimum degree reordering, and its R factor)
       3. D using the metis reordering, and its R factor)
       4. D using the nesdis reordering, and its R factor)
       5. D using the colamd reordering, and its R factor)
       6. D using the 'best' reordering, and its R factor)
    """

    # get size of D
    m,n = D.shape

    # figure 1: original structure
    factor = cholesky(D, ordering_method='natural')
    R = factor.L()
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(D)
    axarr[0].set_title('Original matrix (nnz = ' + f"{D.getnnz():,}" + ')')
    axarr[1].spy(R)
    axarr[1].set_title('Original: R (nnz = ' + f"{R.getnnz():,}" + ')')


    # figure 2: approximate minimum degree
    factor = cholesky(D, ordering_method='amd')
    R = factor.L()
    P = factor.P()
    PD = D[P[:, np.newaxis], P[np.newaxis, :]]
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('AMD matrix (nnz = ' + f"{PD.getnnz():,}" + ')')
    axarr[1].spy(R)
    axarr[1].set_title('AMD: R (nnz = ' + f"{R.getnnz():,}" + ')')

    # figure 3: metis
    factor = cholesky(D, ordering_method='metis')
    R = factor.L()
    P = factor.P()
    PD = D[P[:, np.newaxis], P[np.newaxis, :]]
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('Metis matrix (nnz = ' + f"{PD.getnnz():,}" + ')')
    axarr[1].spy(R)
    axarr[1].set_title('Metis: R (nnz = ' + f"{R.getnnz():,}" + ')')

    # figure 4: nesdis
    factor = cholesky(D, ordering_method='nesdis')
    R = factor.L()
    P = factor.P()
    PD = D[P[:, np.newaxis], P[np.newaxis, :]]
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('nesdis matrix (nnz = ' + f"{PD.getnnz():,}" + ')')
    axarr[1].spy(R)
    axarr[1].set_title('nesdis: R (nnz = ' + f"{R.getnnz():,}" + ')')

    # figure 5: colamd
    factor = cholesky(D, ordering_method='colamd')
    R = factor.L()
    P = factor.P()
    PD = D[P[:, np.newaxis], P[np.newaxis, :]]
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('Colamd matrix (nnz = ' + f"{PD.getnnz():,}" + ')')
    axarr[1].spy(R)
    axarr[1].set_title('Colamd: R (nnz = ' + f"{R.getnnz():,}" + ')')

    # figure 6: best
    factor = cholesky(D, ordering_method='best')
    R = factor.L()
    P = factor.P()
    PD = D[P[:, np.newaxis], P[np.newaxis, :]]
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('Best matrix (nnz = ' + f"{PD.getnnz():,}" + ')')
    axarr[1].spy(R)
    axarr[1].set_title('Best: R (nnz = ' + f"{R.getnnz():,}" + ')')

    plt.show()



def diff_2D(Nx,Ny):
    r"""
    Usage: D = diff_2D(Nx,Ny)

    This routine creates the diffusion matrix resulting from the equation
    \[
         u - \Delta u,
    \]
    where $u \in \Real$ is defined on the square domain [0,1] x [0,1], which
    is discretized using Nx points in the x-direction, and Ny points in the
    y-direction, and the Laplace operator is discretized using the standard
    2nd-order 5 point stencil.  Homogeneous Dirichlet boundary conditions are
    assumed just outside the domain.

    inputs:
        Nx       # spatial points in the x-direction of the domain
        Ny       # spatial points in the y-direction of the domain

    outputs:
        D        REAL (Nx*Ny) x (Nx*Ny) sparse (CSC) matrix
    """

    # set indexing function from 2D physical space to 1D index space
    def ij(i,j):
        return(j*Nx + i)

    # initialize the data and row/column index arrays
    nnzmax = Nx*Ny*5
    rows = np.zeros(nnzmax, dtype=float)
    cols = np.zeros(nnzmax, dtype=float)
    vals = np.zeros(nnzmax, dtype=float)

    # set differencing constants
    dx = 1.0/(Nx-1)
    dy = 1.0/(Ny-1)
    Dx2i = 1.0/dx/dx
    Dy2i = 1.0/dy/dy
    Diag = 1.0 + 2.0*(Dx2i + Dy2i)

    # iterate over the domain
    idx = 0
    for iy in range(Ny):
        for ix in range(Nx):

            # set the matrix entries for this row of D
            #   diagonal
            rows[idx] = ij(ix,iy)
            cols[idx] = ij(ix,iy)
            vals[idx] = Diag
            idx += 1

            #   x-left
            if (ix > 0):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix-1,iy)
                vals[idx] = -Dx2i
                idx += 1

            #   x-right
            if (ix < Nx-1):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix+1,iy)
                vals[idx] = -Dx2i
                idx += 1

            #   y-left
            if (iy > 0):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix,iy-1)
                vals[idx] = -Dy2i
                idx += 1

            #   y-right
            if (iy < Ny-1):
                rows[idx] = ij(ix,iy)
                cols[idx] = ij(ix,iy+1)
                vals[idx] = -Dy2i
                idx += 1

    Dcsc = csc_matrix(coo_matrix((vals, (rows,cols)), shape=(Nx*Ny, Nx*Ny)))
    return Dcsc


def diff_3D(Nx,Ny,Nz):
    r"""
    Usage: D = diff_3D(Nx,Ny,Nz)

    This routine creates the diffusion matrix resulting from the equation
    \[
         u - \Delta u,
    \]
    where $u \in \Real$ is defined on the cube domain [0,1] x [0,1] x [0,1],
    which is discretized using Nx points in the x-direction, Ny points in the
    y-direction, Nz points in the z-direction, and the Laplace operator is
    discretized using the standard 2nd-order 7 point stencil.  Homogeneous
    Dirichlet boundary conditions are assumed just outside the domain.

    inputs:
        Nx       # spatial points in the x-direction of the domain
        Ny       # spatial points in the y-direction of the domain
        Nz       # spatial points in the z-direction of the domain

    outputs:
        D        REAL (Nx*Ny*Nz) x (Nx*Ny*Nz) sparse (CSC) matrix
    """

    # set indexing function from 3D physical space to 1D index space
    def ijk(i,j,k):
        return (k*Nx*Ny + j*Nx + i)

    # initialize the data and row/column index arrays
    nnzmax = Nx*Ny*Nz*7
    rows = np.zeros(nnzmax, dtype=float)
    cols = np.zeros(nnzmax, dtype=float)
    vals = np.zeros(nnzmax, dtype=float)

    # set differencing constants
    dx = 1.0/(Nx-1)
    dy = 1.0/(Ny-1)
    dz = 1.0/(Nz-1)
    Dx2i = 1.0/dx/dx
    Dy2i = 1.0/dy/dy
    Dz2i = 1.0/dz/dz
    Diag = 1.0 + 2.0*(Dx2i + Dy2i + Dz2i)

    # iterate over the domain
    idx = 0
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):

                # set the matrix entries for this row of D

                #   diagonal
                rows[idx] = ijk(ix,iy,iz)
                cols[idx] = ijk(ix,iy,iz)
                vals[idx] = Diag
                idx += 1

                #   x-left
                if (ix > 0):
                    rows[idx] = ijk(ix,iy,iz)
                    cols[idx] = ijk(ix-1,iy,iz)
                    vals[idx] = -Dx2i
                    idx += 1

                #   x-right
                if (ix < Nx-1):
                    rows[idx] = ijk(ix,iy,iz)
                    cols[idx] = ijk(ix+1,iy,iz)
                    vals[idx] = -Dx2i
                    idx += 1

                #   y-left
                if (iy > 0):
                    rows[idx] = ijk(ix,iy,iz)
                    cols[idx] = ijk(ix,iy-1,iz)
                    vals[idx] = -Dy2i
                    idx += 1

                #   y-right
                if (iy < Ny-1):
                    rows[idx] = ijk(ix,iy,iz)
                    cols[idx] = ijk(ix,iy+1,iz)
                    vals[idx] = -Dy2i
                    idx += 1

                #   z-left
                if (iz > 0):
                    rows[idx] = ijk(ix,iy,iz)
                    cols[idx] = ijk(ix,iy,iz-1)
                    vals[idx] = -Dz2i
                    idx += 1

                #   z-right
                if (iz < Nz-1):
                    rows[idx] = ijk(ix,iy,iz)
                    cols[idx] = ijk(ix,iy,iz+1)
                    vals[idx] = -Dz2i
                    idx += 1


    Dcsc = csc_matrix(coo_matrix((vals, (rows,cols)), shape=(Nx*Ny*Nz, Nx*Ny*Nz)))
    return Dcsc



##################
# script

if __name__ == "__main__":

    # problem 1: small 2D diffusion matrix
    print("problem 1: small 2D diffusion matrix")
    D = diff_2D(5,10)
    makeplots(D)

    # problem 2: larger 2D diffusion matrix
    print("problem 2: larger 2D diffusion matrix");
    D = diff_2D(50,100)
    makeplots(D)

    # problem 3: small 3D diffusion matrix
    print("problem 3: small 3D diffusion matrix");
    D = diff_3D(5,8,10)
    makeplots(D)

    # problem 4: larger 3D diffusion matrix
    print("problem 4: larger 3D diffusion matrix");
    D = diff_3D(20,25,30)
    makeplots(D)

#!/usr/bin/env python3
#
# Utility functions to create diffusion matrices resulting from the equation
# $$u - \Delta u,$$
# where $u \in \Real$ is defined on either the 2D square domain [0,1] x [0,1]
# or the 3D cubic domain [0,1]^3, which is discretized using Nx points in 
# the x-direction, Ny points in the y-direction, and for the 3D version uses 
# Nz points in the z-direction.  The Laplace operator is discretized using 
# the standard 2nd-order 5 point or 7 point stencils in 2D and 3D, respectively.
# Homogeneous Dirichlet boundary conditions are assumed just outside the domain.
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

def diff_2D(Nx,Ny):
    r"""
    Usage: D = diff_2D(Nx,Ny)

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

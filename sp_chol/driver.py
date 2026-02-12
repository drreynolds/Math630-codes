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
import diff
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


##################
# script

if __name__ == "__main__":

    # problem 1: small 2D diffusion matrix
    print("problem 1: small 2D diffusion matrix")
    D = diff.diff_2D(5,10)
    makeplots(D)

    # problem 2: larger 2D diffusion matrix
    print("problem 2: larger 2D diffusion matrix");
    D = diff.diff_2D(50,100)
    makeplots(D)

    # problem 3: small 3D diffusion matrix
    print("problem 3: small 3D diffusion matrix");
    D = diff.diff_3D(5,8,10)
    makeplots(D)

    # problem 4: larger 3D diffusion matrix
    print("problem 4: larger 3D diffusion matrix");
    D = diff.diff_3D(20,25,30)
    makeplots(D)

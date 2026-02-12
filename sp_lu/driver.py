#!/usr/bin/env python3
#
# Script to demonstrate sparse LU factorizations (and reorderings).
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import time
import numpy as np
import diff
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt


##################
# utility routines

def makeplots(D):
    """
    Usage: makeplots(D)

    This routine creates 3 plots:
       1. D and its L,U factors
       2. D using the approximate minimum degree reordering on D.T @ D, and its
          L,U factors)
       3. D using the approximate minimum degree reordering on D.T + D, and its
          L,U factors)
       4. D using the approximate minimum degree on column ordering, and its L,U
       factors)
    """

    # get size of D
    m,n = D.shape

    # figure 1: original structure and L,U factors
    lu = la.splu(D, permc_spec='NATURAL')
    LU = lu.L + lu.U
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(D)
    axarr[0].set_title('Original matrix (nnz = ' + f"{D.getnnz():,}" + ')')
    axarr[1].spy(LU)
    axarr[1].set_title('Original: L+U (nnz = ' + f"{LU.nnz:,}" + ')')


    # figure 2: approximate minimum degree on D.T@D
    lu = la.splu(D, permc_spec='MMD_ATA')
    LU = lu.L + lu.U
    PD = csc_matrix(lu.L @ lu.U)
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('Approximate minimum degree on A.T @ A (nnz = ' + f"{PD.nnz:,}" + ')')
    axarr[1].spy(LU)
    axarr[1].set_title('AMD A.T @ A: L+U (nnz = ' + f"{LU.nnz:,}" + ')')


    # figure 3: approximate minimum degree on A.T+A
    lu = la.splu(D, permc_spec='MMD_AT_PLUS_A')
    LU = lu.L + lu.U
    PD = csc_matrix(lu.L @ lu.U)
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('Approximate minimum degree on A.T + A (nnz = ' + f"{PD.nnz:,}" + ')')
    axarr[1].spy(LU)
    axarr[1].set_title('AMD A.T + A: L+U (nnz = ' + f"{LU.nnz:,}" + ')')


    # figure 4: approximate minimum degree on column ordering
    lu = la.splu(D, permc_spec='COLAMD')
    LU = lu.L + lu.U
    PD = csc_matrix(lu.L @ lu.U)
    fig, axarr = plt.subplots(1,2)
    axarr[0].spy(PD)
    axarr[0].set_title('Approximate minimum degree column order (nnz = ' + f"{PD.nnz:,}" + ')')
    axarr[1].spy(LU)
    axarr[1].set_title('COLAMD: L+U (nnz = ' + f"{LU.nnz:,}" + ')')

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

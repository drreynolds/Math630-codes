#!/usr/bin/env python3
#
# Script to test QRfact on a variety of matrices.
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import numpy as np
import warnings
from QRfact import QRfact
from scipy import linalg

# suppress runtime warnings arising from built-in matrix-matrix products
warnings.filterwarnings("ignore", category=RuntimeWarning)

# set matrix sizes for tests
nvals = [50, 100, 200, 400]

# full-rank square matrix tests
for n in nvals:

    print("Testing with full-rank square matrix of dimension ", n)

    # create the matrix
    I = np.eye(n)
    A = np.random.rand(n,n) + I

    # call QRfact
    Q, R = QRfact(A)

    # output results
    print("   ||I-Q^TQ||     = ", linalg.norm(I-np.transpose(Q)@Q,2))
    print("   ||I-QQ^T||     = ", linalg.norm(I-Q@np.transpose(Q),2))
    print("   ||A-QR||       = ", linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", linalg.norm(np.tril(R,-1),2))

# full-rank rectangular matrix tests
for n in nvals:

    print("Testing with full-rank rectangular matrix of dimension ", 2*n, "x", n)

    # create the matrix
    I = np.eye(2*n)
    A = np.random.rand(2*n,n) + I[:,:n]

    # call QRfact
    Q, R = QRfact(A)

    # output results
    print("   ||I-Q^TQ||     = ", linalg.norm(I-np.transpose(Q)@Q,2))
    print("   ||I-QQ^T||     = ", linalg.norm(I-Q@np.transpose(Q),2))
    print("   ||A-QR||       = ", linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", linalg.norm(np.tril(R,-1),2))

# rank-deficient square matrix tests
for n in nvals:

    print("Testing with rank-deficient square matrix of dimension ", n)

    # create the matrix
    I = np.eye(n)
    A = np.random.rand(n,n) + I
    A[:,2] = 2*A[:,1]

    # call QRfact
    Q, R = QRfact(A)

    # output results
    print("   ||I-Q^TQ||     = ", linalg.norm(I-np.transpose(Q)@Q,2))
    print("   ||I-QQ^T||     = ", linalg.norm(I-Q@np.transpose(Q),2))
    print("   ||A-QR||       = ", linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", linalg.norm(np.tril(R,-1),2))

# rank-deficient rectangular matrix tests
for n in nvals:

    print("Testing with rank-deficient rectangular matrix of dimension ", 2*n, "x", n)

    # create the matrix
    I = np.eye(2*n)
    A = np.random.rand(2*n,n) + I[:,:n]

    # call QRfact
    Q, R = QRfact(A)

    # output results
    print("   ||I-Q^TQ||     = ", linalg.norm(I-np.transpose(Q)@Q,2))
    print("   ||I-QQ^T||     = ", linalg.norm(I-Q@np.transpose(Q),2))
    print("   ||A-QR||       = ", linalg.norm(A-Q@R,2))
    print("   ||tril(R,-1)|| = ", linalg.norm(np.tril(R,-1),2))


# end of script

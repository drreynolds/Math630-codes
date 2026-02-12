#!/usr/bin/env python3
#
# Utility routines to perform stationary iterations on our 2D Laplace problem
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026

# imports
import numpy as np
from laplace_2D import laplace_2D

##################
# utility routines

def jacobi(u, b):
    """perform one Jacobi iteration for our test problem"""
    N = int(np.sqrt(b.size))
    def ij(i,j):
        return j*N + i
    h = 1.0/(N+1)
    h2 = h*h
    u_new = np.zeros_like(u)
    for j in range(N):
        for i in range(N):
            u_new[ij(i,j)] = 0.25*b[ij(i,j)]
            if (i > 0):
                u_new[ij(i,j)] += 0.25*u[ij(i-1,j)]
            if (i < N-1):
                u_new[ij(i,j)] += 0.25*u[ij(i+1,j)]
            if (j > 0):
                u_new[ij(i,j)] += 0.25*u[ij(i,j-1)]
            if (j < N-1):
                u_new[ij(i,j)] += 0.25*u[ij(i,j+1)]
    return u_new

def fwd_gauss_seidel(u, b):
    """perform one forward-sweep Gauss-Seidel iteration for our test problem"""
    N = int(np.sqrt(b.size))
    def ij(i,j):
        return j*N + i
    h = 1.0/(N+1)
    h2 = h*h
    for j in range(N):
        for i in range(N):
            u[ij(i,j)] = 0.25*b[ij(i,j)]
            if (i > 0):
                u[ij(i,j)] += 0.25*u[ij(i-1,j)]
            if (i < N-1):
                u[ij(i,j)] += 0.25*u[ij(i+1,j)]
            if (j > 0):
                u[ij(i,j)] += 0.25*u[ij(i,j-1)]
            if (j < N-1):
                u[ij(i,j)] += 0.25*u[ij(i,j+1)]
    return u  

def bwd_gauss_seidel(u, b):
    """perform one backward-sweep Gauss-Seidel iteration for our test problem"""
    N = int(np.sqrt(b.size))
    def ij(i,j):
        return j*N + i
    h = 1.0/(N+1)
    h2 = h*h
    for j in range(N-1,-1,-1):
        for i in range(N-1,-1,-1):
            u[ij(i,j)] = 0.25*b[ij(i,j)]
            if (i > 0):
                u[ij(i,j)] += 0.25*u[ij(i-1,j)]
            if (i < N-1):
                u[ij(i,j)] += 0.25*u[ij(i+1,j)]
            if (j > 0):
                u[ij(i,j)] += 0.25*u[ij(i,j-1)]
            if (j < N-1):
                u[ij(i,j)] += 0.25*u[ij(i,j+1)]
    return u    

def red_black_gauss_seidel(u, b):
    """perform one iteration of Red-Black Gauss-Seidel for our test problem"""
    N = int(np.sqrt(b.size))
    def ij(i,j):
        return j*N + i
    h = 1.0/(N+1)
    h2 = h*h
    for j in range(N):
        for i in range(N):
            if ((i+j) % 2 == 0):
                continue
            u[ij(i,j)] = 0.25*b[ij(i,j)]
            if (i > 0):
                u[ij(i,j)] += 0.25*u[ij(i-1,j)]
            if (i < N-1):
                u[ij(i,j)] += 0.25*u[ij(i+1,j)]
            if (j > 0):
                u[ij(i,j)] += 0.25*u[ij(i,j-1)]
            if (j < N-1):
                u[ij(i,j)] += 0.25*u[ij(i,j+1)]
    for j in range(N):
        for i in range(N):
            if ((i+j) % 2 == 1):
                continue
            u[ij(i,j)] = 0.25*b[ij(i,j)]
            if (i > 0):
                u[ij(i,j)] += 0.25*u[ij(i-1,j)]
            if (i < N-1):
                u[ij(i,j)] += 0.25*u[ij(i+1,j)]
            if (j > 0):
                u[ij(i,j)] += 0.25*u[ij(i,j-1)]
            if (j < N-1):
                u[ij(i,j)] += 0.25*u[ij(i,j+1)]
    return u    

def symmetric_gauss_seidel(u, b):
    """perform one symmetric Gauss-Seidel iteration for our test problem"""
    u = fwd_gauss_seidel(u, b)
    u = bwd_gauss_seidel(u, b)
    return u  

def sor(u, b, omega):
    """perform one SOR iteration for our test problem"""
    N = int(np.sqrt(b.size))
    def ij(i,j):
        return j*N + i
    h = 1.0/(N+1)
    h2 = h*h
    for j in range(N):
        for i in range(N):
            uhat = 0.25*b[ij(i,j)]
            if (i > 0):
                uhat += 0.25*u[ij(i-1,j)]
            if (i < N-1):
                uhat += 0.25*u[ij(i+1,j)]
            if (j > 0):
                uhat += 0.25*u[ij(i,j-1)]
            if (j < N-1):
                uhat += 0.25*u[ij(i,j+1)]
            u[ij(i,j)] = (1.0 - omega)*u[ij(i,j)] + omega*uhat
    return u  

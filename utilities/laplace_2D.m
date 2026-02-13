function [A,b] = laplace_2D(M, f, g)
% Usage: [A,b] = laplace_2D(M, f, g)
%
% This routine creates a sparse matrix and right-hand side corresponding 
% to the linear system
% $$-\Delta u(x,y) = f(x,y), \quad (x,y) \in [0,1]^2$$
% with boundary conditions
% $$u(x,y) = g(x,y), \quad (x,y) \in \partial [0,1]^2$$
% where $u \in \Real$ is discretized using Nx points in the x-direction and 
% Ny points in the y-direction.  The Laplace operator is discretized using 
% the standard 2nd-order 5 point stencil, and the boundary condition 
% function is passed in as an input.
%
% inputs:
%     M    # intervals in each of the x and y directions
%     f    # forcing function
%     g    # boundary condition function
%
% outputs:
%     A    REAL ((M-1)*(M-1)) x ((M-1)*(M-1)) sparse (CSC) matrix
%     b    REAL ((M-1)*(M-1)) vector
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

% set number of interior finite difference points in each direction
N = M - 1;

% set indexing function from 2D physical space to 1D index space
ij = @(i,j) (j-1)*N + i;

% initialize the output matrix
nrows = N*N;
A = sparse(nrows,nrows);

% initialize the RHS vector b
b = zeros(nrows,1);

% set differencing constants
h = 1.0/M;
h2 = h*h;

% iterate over the domain
for iy=1:N
   for ix=1:N

      % set the x,y location and neighboring points
      x = ix*h;
      y = iy*h;

      % set the forcing term into the RHS
      b(ij(ix,iy)) = f(x, y)*h2;

      % set the matrix entries for this row of D
      A( ij(ix,iy), ij(ix,iy) ) = 4.0;
      if (ix > 1)
	      A( ij(ix,iy), ij(ix-1,iy) ) = -1.0;
      else
         b(ij(ix,iy)) = b(ij(ix,iy)) + g(0.0,y);
      end
      if (ix < N)
	      A( ij(ix,iy), ij(ix+1,iy) ) = -1.0;
      else
         b(ij(ix,iy)) = b(ij(ix,iy)) + g(1.0,y);
      end
      if (iy > 1)
	      A( ij(ix,iy), ij(ix,iy-1) ) = -1.0;
      else
         b(ij(ix,iy)) = b(ij(ix,iy)) + g(x,0.0);
      end
      if (iy < N)
	      A( ij(ix,iy), ij(ix,iy+1) ) = -1.0;
      else
         b(ij(ix,iy)) = b(ij(ix,iy)) + g(x,1.0);
      end

   end
end


% end of function
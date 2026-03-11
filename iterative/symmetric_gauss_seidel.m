function u = symmetric_gauss_seidel(u, b)
% Usage: u = symmetric_gauss_seidel(u, b)
%
% Function to perform one symmetric Gauss-Seidel iteration for our test problem
%
% Inputs:
%    u is the current guess vector (n)
%    b is the linear system right-hand side vector (n)
%
% Outputs:
%    u is the updated guess vector (n)
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

% symmetric Gauss-Seidel just consists of two passes (one in each direction)
u = fwd_gauss_seidel(u, b);
u = bwd_gauss_seidel(u, b);

% end of function

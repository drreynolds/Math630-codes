function [x,iters,errnorm] = steepest_descent(A, Pinv, x, b, maxit, tol)
% Usage: [x,iters,errnorm] = steepest_descent(A, Pinv, x, b, maxit, tol)
%
% Function to perform preconditioned steepest descent
%
% Inputs:
%    A is the sparse matrix (n x n)
%    Pinv is a function for the preconditioner solve, 
%         i.e. to y = Pinv(z) computes y = P^{-1}@z 
%         To run without a preconditioner, supply "0"
%    x is the initial guess vector (n)
%    b is the linear system right-hand side vector (n)
%    maxit is the maximum number of allowed iterations
%    tol is the requested relative solution tolerance
%
% Outputs:
%    x is the final iterate vector (n)
%    iters is the number of iterations performed
%    errnorm is ||x-x0||/||x||
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

% if Pinv==0, supply a dummy
if ~isa(Pinv, 'function_handle')
    Pinv = @(z) z;
end

% perform algorithm
r = b - A*x;
p = Pinv(r);
for iters = 1:maxit
    q = A*p;
    alpha = dot(p,r) / dot(p,q);
    x = x + alpha*p;
    r = r - alpha*q;
    dxnorm = abs(alpha)*norm(p);
    xnorm = norm(x);
    p = Pinv(r);
    if (dxnorm <= xnorm*tol)
        break;
    end
end
errnorm = dxnorm/xnorm;

% end of function

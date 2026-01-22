function B = matmat_kij(A,X)
% Usage: B = matmat_kij(A,X)
%
% Function to perform matrix-matrix multiplication with "kij" loop ordering.
%
% Inputs: A is a matrix (m x n numpy matrix)
%         X is a matrix (n x p numpy matrix)
% Output: B is a matrix (m x p numpy matrix)
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

% get problem dimensions
[m,n]  = size(A);
[n2,p] = size(X);

% check that A and X are compatible
if (n ~= n2)
   error('matmat_kij error: A and X are incompatible')
end

% initialize output
B = zeros(m,p);

% perform product
for k=1:n
   for i=1:m
      for j=1:p
         B(i,j) = B(i,j) + A(i,k)*X(k,j);
      end
   end
end

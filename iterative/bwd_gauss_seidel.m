function u = bwd_gauss_seidel(u, b)
% Usage: u = bwd_gauss_seidel(u, b)
%
% Function to perform one backward-sweep Gauss-Seidel iteration for our test problem
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

% deduce mesh size from RHS vector
N = sqrt(length(b));

% set function to help map from 2D logical space to 1D index space
ij = @(i,j) (j-1)*N + i;

% loop over physical domain in reverse order, implementing formula (8.2.16) from the book
% note that we modify the input vector in-place instead of filling a separate output
for j = N:-1:1
    for i = N:-1:1
        
        % b holds h^2*f_{i,j}, along with modifications for boundary data
        u(ij(i,j)) = 0.25*b(ij(i,j));

        % implement update at (i,j) location, accounting for modifications at boundaries
        if (i > 1)
            u(ij(i,j)) = u(ij(i,j)) + 0.25*u(ij(i-1,j));
        end
        if (i < N)
            u(ij(i,j)) = u(ij(i,j)) + 0.25*u(ij(i+1,j));
        end
        if (j > 1)
            u(ij(i,j)) = u(ij(i,j)) + 0.25*u(ij(i,j-1));
        end
        if (j < N)
            u(ij(i,j)) = u(ij(i,j)) + 0.25*u(ij(i,j+1));
        end
    end
end

% end of function

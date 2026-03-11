function u_new = jacobi(u, b)
% Usage: u_new = jacobi(u, b)
%
% Function to perform one Jacobi iteration for our test problem
%
% Inputs:
%    u is the current guess vector (n)
%    b is the linear system right-hand side vector (n)
%
% Outputs:
%    u_new is the updated guess vector (n)
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

% deduce mesh size from RHS vector
N = sqrt(length(b));

% set function to help map from 2D logical space to 1D index space
ij = @(i,j) (j-1)*N + i;

% initialize output
u_new = zeros(size(u));

% loop over physical domain, implementing formula from the bottom of page 556 of book
for j = 1:N
    for i = 1:N
        % b holds h^2*f_{i,j}, along with modifications for boundary data
        u_new(ij(i,j)) = 0.25*b(ij(i,j));

        % implement update at (i,j) location, accounting for modifications at boundaries
        if (i > 1)  
            u_new(ij(i,j)) = u_new(ij(i,j)) + 0.25*u(ij(i-1,j));
        end
        if (i < N)
            u_new(ij(i,j)) = u_new(ij(i,j)) + 0.25*u(ij(i+1,j));
        end
        if (j > 1)
            u_new(ij(i,j)) = u_new(ij(i,j)) + 0.25*u(ij(i,j-1));
        end
        if (j < N)
            u_new(ij(i,j)) = u_new(ij(i,j)) + 0.25*u(ij(i,j+1));
        end
    end
end

% end of function

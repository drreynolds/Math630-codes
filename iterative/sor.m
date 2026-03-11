function u = sor(u, b, omega)
% Usage: u = sor(u, b, omega)
%
% Function to perform one SOR iteration for our test problem
%
% Inputs:
%    u is the current guess vector (n)
%    b is the linear system right-hand side vector (n)
%    omega is the SOR relaxation parameter (scalar)
%
% Outputs:
%    u is the updated guess vector (n)
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

N = sqrt(length(b));
ij = @(i,j) (j-1)*N + i;
for j = 1:N
    for i = 1:N
        uhat = 0.25*b(ij(i,j));
        if (i > 1)
            uhat = uhat + 0.25*u(ij(i-1,j));
        end
        if (i < N)
            uhat = uhat + 0.25*u(ij(i+1,j));
        end
        if (j > 1)
            uhat = uhat + 0.25*u(ij(i,j-1));
        end
        if (j < N)
            uhat = uhat + 0.25*u(ij(i,j+1));
        end
        u(ij(i,j)) = (1.0 - omega)*u(ij(i,j)) + omega*uhat;
    end
end

% end of function

function u = red_black_gauss_seidel(u, b)
% Usage: u = red_black_gauss_seidel(u, b)
%
% Function to perform one Red-Black Gauss-Seidel iteration for our test problem
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

N = sqrt(length(b));
ij = @(i,j) (j-1)*N + i;
h = 1/(N+1);
for j = 1:N
    for i = 1:N
        if (mod(i+j,2) == 0)
            continue
        end
        u(ij(i,j)) = 0.25*b(ij(i,j));
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
for j = 1:N
    for i = 1:N
        if (mod(i+j,2) == 1)
            continue
        end
        u(ij(i,j)) = 0.25*b(ij(i,j));
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

% Script to test simple iterative methods on our 2D Laplace problem
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

clear

% utility routines
f = @(x,y) 0.0;
initial_guess = @(M) zeros((M-1)*(M-1),1);
% the boundary condition function g is included at the end of the file

%%%%%%%%%%%%%%
% tests

% set grid sizes to try
Mvals = [20,40,80];

% set stopping tolerance
tol = 1.0e-6;

% set maximum allowed number of iterations
maxiters = 10000;

% set SOR relaxation parameter values to try
omega_vals = [0.8, 1.0, 1.5, 1.8, 1.9, 1.95, 1.97, 2.0];

% loop over problem sizes, running each iterative solver
for M = Mvals

    fprintf("Problem size = %d\n", M)
    [A,b] = laplace_2D(M,f,@g);
    xtrue = A\b;

    % run Jacobi method
    x0 = initial_guess(M);
    stime = tic;
    for niters = 1:maxiters
        x = x0;
        x = jacobi(x, b);
        % break when relative change in iterates is below tol
        if (norm(x - x0)/norm(x) < tol)
            break
        end
        x0 = x;
    end
    runtime = toc(stime);
    fprintf("  Jacobi:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run [forward] Gauss-Seidel method
    x0 = initial_guess(M);
    stime = tic;
    for niters = 1:maxiters
        x = x0;
        x = fwd_gauss_seidel(x, b);
        if (norm(x - x0)/norm(x) < tol)
            break
        end
        x0 = x;
    end
    runtime = toc(stime);
    fprintf("  GS:        niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run Red-Black Gauss-Seidel method
    x0 = initial_guess(M);
    stime = tic;
    for niters = 1:maxiters
        x = x0;
        x = red_black_gauss_seidel(x, b);
        if (norm(x - x0)/norm(x) < tol)
            break
        end
        x0 = x;
    end
    runtime = toc(stime);
    fprintf("  RB-GS:     niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run Symmetric Gauss-Seidel method
    x0 = initial_guess(M);
    stime = tic;
    for niters = 1:maxiters
        x = x0;
        x = symmetric_gauss_seidel(x, b);
        if (norm(x - x0)/norm(x) < tol)
            break
        end
        x0 = x;
    end
    runtime = toc(stime);
    fprintf("  Sym-GS:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            2*niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run SOR method for each omega value
    for omega = omega_vals
        x0 = initial_guess(M);
        stime = tic;
        for niters = 1:maxiters
            x = x0;
            x = sor(x, b, omega);
            if (norm(x - x0)/norm(x) < tol)
                break
            end
            x0 = x;
        end
        runtime = toc(stime);
        fprintf("  SOR(%.2f): niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
                omega, niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    end

end  % loop over problem sizes


% boundary condition function
function gval = g(x,y)
    tol = sqrt(eps);
    gval = 0.0;
    if (abs(x-0.0) < tol)
        gval = 0.0;
    elseif (abs(x-1.0) < tol)
        gval = y;
    elseif (abs(y-0.0) < tol)
        gval = (x-1.0)*sin(x);
    elseif (abs(y-1.0) < tol)
        gval = x*(2.0-x);
    end
end

% end of script

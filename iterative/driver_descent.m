% Script to test descent methods on our 2D Laplace problem
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
Mvals = [20,40,80,160,320];

% set stopping tolerance
tol = 1.0e-6;

% set maximum allowed number of iterations
maxiters = 10000;

% loop over problem sizes, running each iterative solver
for M = Mvals

    fprintf("Problem size = %d\n", M)
    [A,b] = laplace_2D(M,f,@g);
    xtrue = A\b;

    % create preconditioners
    D = diag(diag(A));
    E = -tril(A,-1);
    F = -triu(A,1);
    % Jacobi: M = D, so P = D^{-1}
    PJacobi = @(x) D\x;
    % symmetric GS: M = L D^{-1} U, where L=D-E, U=D-F
    %   so P = U^{-1} D L^{-1}
    L = D-E;
    U = D-F;
    PSGS = @(x) U\(D*(L\x));
    % SSOR: M = om/(2-om)*L*D^{-1}*U, where L=1/om*D-E, U=1/om*D-F
    %   so P = (2-om)/om*U^{-1}*D*L^{-1}
    om = 1.5;
    L = (1/om)*D-E;
    U = (1/om)*D-F;
    PSOR1 = @(x) ((2-om)/om)*(U\(D*(L\x)));
    om = 1.9;
    L = (1/om)*D-E;
    U = (1/om)*D-F;
    PSOR2 = @(x) ((2-om)/om)*(U\(D*(L\x)));

    % run un-preconditioned steepest descent
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = steepest_descent(A, 0, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  SD:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run steepest descent with Jacobi preconditioning
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = steepest_descent(A, PJacobi, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  SD-Jac:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run steepest descent with symmetric Gauss-Seidel preconditioning
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = steepest_descent(A, PSGS, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  SD-SGS:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run steepest descent with SSOR preconditioner 1
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = steepest_descent(A, PSOR1, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  SD-SOR1:  niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run steepest descent with SSOR preconditioner 2
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = steepest_descent(A, PSOR2, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  SD-SOR2:  niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run un-preconditioned conjugate gradient
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = conjugate_gradient(A, 0, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  CG:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run conjugate gradient with Jacobi preconditioning
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = conjugate_gradient(A, PJacobi, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  CG-Jac:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run conjugate gradient with symmetric Gauss-Seidel preconditioning
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = conjugate_gradient(A, PSGS, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  CG-SGS:   niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run conjugate gradient with SOR preconditioner 1
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = conjugate_gradient(A, PSOR1, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  CG-SOR1:  niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run conjugate gradient with SOR preconditioner 2
    x0 = initial_guess(M);
    stime = tic;
    [x,niters,~] = conjugate_gradient(A, PSOR2, x0, b, maxiters, tol);
    runtime = toc(stime);
    fprintf("  CG-SOR2:  niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

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

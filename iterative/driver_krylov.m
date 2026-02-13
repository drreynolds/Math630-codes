% Script to test built-in Krylov on our 2D Laplace problem
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

clear

% utility routines
f = @(x,y) 0.0;
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

    % create preconditioner
    D = diag(diag(A));
    E = -tril(A,-1);
    F = -triu(A,1);
    % SSOR: M = om/(2-om)*L*D^{-1}*U, where L=1/om*D-E, U=1/om*D-F
    %   so P = (2-om)/om*U^{-1}*D*L^{-1}
    om = 1.5;
    L = (1/om)*D-E;
    U = (1/om)*D-F;
    P = @(x,opt) ((2-om)/om)*(U\(D*(L\x)));

    % run pcg
    stime = tic;
    [x,~,~,niters] = pcg(A, b, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  pcg:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run minres
    stime = tic;
    [x,~,~,niters] = minres(A, b, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  minres:    niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run gmres with restart size of 10
    stime = tic;
    [x,~,~,niters] = gmres(A, b, 10, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  gmres(10): niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters(1)*10+niters(2), runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run gmres with restart size of 20
    stime = tic;
    [x,~,~,niters] = gmres(A, b, 20, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  gmres(20): niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters(1)*20+niters(2), runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run bicg
    stime = tic;
    [x,~,~,niters] = bicg(A, b, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  bicg:      niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run cgs
    stime = tic;
    [x,~,~,niters] = cgs(A, b, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  cgs:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            niters, runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run bicgstab
    stime = tic;
    [x,~,~,niters] = bicgstab(A, b, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  bicgstab:  niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
            ceil(niters), runtime, norm(x - xtrue)/norm(xtrue), norm(b - A*x)/norm(b));

    % run qmr
    stime = tic;
    [x,~,~,niters] = qmr(A, b, tol, maxiters, P);
    runtime = toc(stime);
    fprintf("  qmr:       niters = %5d, time = %.1e, error = %.1e, residual = %.1e\n", ...
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

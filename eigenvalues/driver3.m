% Script to use francis1 and francis2 functions to compute Schur
% decompositions of some matrices.
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

clear

% general solver parameters
maxit = 100;
tol = 1e-6;


% symmetric matrix
n = 10;
A = rand(n,n);
A = A+A';
fprintf('first matrix A (symmetric):\n')
fprintf('  ||A - A''|| = %g\n\n', norm(A-A'));

% convert to upper-Hessenberg form
[H,Q] = upper_hess(A);
fprintf('upper-Hessenberg H = Q''AQ:\n');
fprintf('checks:\n');
fprintf('  norm of lower portion of H = %g\n', norm(H - triu(H,-1)));
fprintf('  ||Q''Q - I|| = %g\n', norm(Q'*Q-eye(n)));
fprintf('  ||QQ'' - I|| = %g\n', norm(Q*Q'-eye(n)));
fprintf('  ||Q''AQ - H|| = %g\n', norm(Q'*A*Q-H));
fprintf('  norm of subdiagonal of H = %g\n\n', norm(diag(H,-1)));
pause


% run Francis1 with Rayleigh quotient shift
fprintf('Running francis1 with Rayleigh quotient shift:\n\n');
[T,U,its] = francis1(H,maxit,tol,0,1);
fprintf('checks:\n');
fprintf('  ||U''U - I|| = %g\n', norm(U'*U-eye(n)));
fprintf('  ||UU'' - I|| = %g\n', norm(U*U'-eye(n)));
fprintf('  ||U''HU - T|| = %g\n', norm(U'*H*U-T));
fprintf('  ||U''Q''AQU - T|| = %g\n', norm(U'*Q'*A*Q*U-T));
fprintf('  norm of lower triangle of T = %g\n\n', norm(tril(T,-1)));
pause

% run Francis1 with Wilkinson shift
fprintf('Running francis1 with Wilkinson shift:\n\n');
[T,U,its] = francis1(H,maxit,tol,1,1);
fprintf('checks:\n');
fprintf('  ||U''U - I|| = %g\n', norm(U'*U-eye(n)));
fprintf('  ||UU'' - I|| = %g\n', norm(U*U'-eye(n)));
fprintf('  ||U''HU - T|| = %g\n', norm(U'*H*U-T));
fprintf('  ||U''Q''AQU - T|| = %g\n', norm(U'*Q'*A*Q*U-T));
fprintf('  norm of lower triangle of T = %g\n\n', norm(tril(T,-1)));
pause



% non-symmetric matrix
n = 10;   % must be even
v = rand(n,n/2) + i*rand(n,n/2);
V = [v, conj(v)];
d = rand(n/2,1) + i*rand(n/2,1);
D = diag([d; conj(d)]);
A = real(V*D*inv(V));
fprintf('\nsecond matrix A (non-symmetric):\n')
fprintf('  ||A - A''|| = %g\n\n', norm(A-A'));

% convert to upper-Hessenberg form
[H,Q] = upper_hess(A);
fprintf('upper-Hessenberg H = Q''AQ:\n');
fprintf('checks:\n');
fprintf('  norm of lower portion of H = %g\n', norm(H - triu(H,-1)));
fprintf('  ||Q''Q - I|| = %g\n', norm(Q'*Q-eye(n)));
fprintf('  ||QQ'' - I|| = %g\n', norm(Q*Q'-eye(n)));
fprintf('  ||Q''AQ - H|| = %g\n', norm(Q'*A*Q-H));
fprintf('  norm of subdiagonal of H = %g\n\n', norm(diag(H,-1)));
pause


% run Francis1 with Rayleigh quotient shift
fprintf('Running francis1 with Rayleigh quotient shift:\n\n');
[T,U,its] = francis1(H,maxit,tol,0,1);
fprintf('checks:\n');
fprintf('  ||U''U - I|| = %g\n', norm(U'*U-eye(n)));
fprintf('  ||UU'' - I|| = %g\n', norm(U*U'-eye(n)));
fprintf('  ||U''HU - T|| = %g\n', norm(U'*H*U-T));
fprintf('  ||U''Q''AQU - T|| = %g\n', norm(U'*Q'*A*Q*U-T));
fprintf('  norm of lower triangle of T = %g\n\n', norm(tril(T,-1)));
pause

% run Francis1 with Wilkinson shift
fprintf('Running francis1 with Wilkinson shift:\n\n');
[T,U,its] = francis1(H,maxit,tol,1,1);
fprintf('checks:\n');
fprintf('  ||U''U - I|| = %g\n', norm(U'*U-eye(n)));
fprintf('  ||UU'' - I|| = %g\n', norm(U*U'-eye(n)));
fprintf('  ||U''HU - T|| = %g\n', norm(U'*H*U-T));
fprintf('  ||U''Q''AQU - T|| = %g\n', norm(U'*Q'*A*Q*U-T));
fprintf('  norm of lower triangle of T = %g\n\n', norm(tril(T,-1)));

function [Q,R] = QRfact(A)
% Usage: [Q,R] = QRfact(A)
%
% Function to compute the QR factorization of a (possibly rank-deficient)
% 'tall-skinny' matrix A (m x n, with m >=n) using Householder
% reflection matrices.
%
% Input:    A - tall-skinny matrix
% Outputs:  Q - orthogonal matrix
%           R - "upper triangular" matrix, i.e. R = [ Rhat ]
%                                                   [  0   ]
%               with Rhat an (n x n) upper-triangular matrix
%
% Daniel R. Reynolds
% Math 630 @ UMBC
% Spring 2026

% get dimensions of A
[m,n] = size(A);

% initialize results
Q = eye(m);
R = A;

% determine elimination extent
if (m==n)
   jend = n-1;
else
   jend = n;
end

% iterate over columns
for j=1:jend

   % extract subvector from diagonal down and compute norm
   x = R(j:m,j);
   tau = norm(x)*sign(x(1));

   % if subvector has norm zero, continue to next column
   if (tau == 0)
      continue
   end

   % compute u = (x-y)/(x_1 + tau) and gamma = 2/||u||^2;
   % the Householder matrix is then Qj = I-gamma*u*u'
   u = x/(x(1)+tau);
   u(1) = 1;
   gamma = (tau+x(1))/tau;

   % update R with [I, 0; 0, Qj]*R
   % this updates only the submatrix R22 = R(j:m, j:n):
   %   [I 0 ] * [R11 R12] = [R11   R12 ]
   %   [0 Qj]   [ 0  R22]   [ 0  Qj*R22]
   % and
   %   Qj*R22 = (I-gamma*u*u')*R22
   %          = R22 - gamma*u*u'*R22
   %          = R22 - u*(gamma*(u'*R22))
   vt = u' * R(j:m, j:n);
   wt = gamma * vt;
   R(j:m, j:n) = R(j:m, j:n) - u * wt;

   % update Q with Q*[I, 0; 0, Qj]
   % this updates only the submatrix Qhat = Q(:,j:m):
   %   [ Q11 Q12 ] * [I, 0 ] = [Q11 Q12*Qj]
   %   [ Q21 Q22 ]   [0, Qj]   [Q21 Q22*Qj  ]
   % or
   %   [ Q(:,1:j-1) Qhat*Qj ]
   % and
   %   Qhat*Qj = Qhat*(I-gamma*u*u')
   %           = Qhat - (gamma*(Qhat*u))*u'
   Qu = Q(:,j:m) * u;
   gQu = gamma * Qu;
   Q(:,j:m) = Q(:,j:m) - gQu * u';

end


% end function
/* Daniel R. Reynolds
   Math 630 @ UMBC
   Spring 2026 */

// Inclusions
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <fstream>
using namespace std;

// function prototypes
void matvec_row(double **A, double *x, double *b, int n, int m);
int cholesky_ip(double **A, int n);
int cholesky_op(double **A, int n);
int fwdsub_row(double **L, double *y, double *b, int n);
int bwdsub_row(double **U, double *x, double *y, int n);


// main driver routine
int main(int argc, char* argv[]) {

  // local variables
  int n, i, j;
  double **A, **R, **Rt, *xtrue, *x, *y, *b, err_norm;
  int nvals[] = {500, 700, 900, 1100};
  int nsizes = 4;
  chrono::time_point<chrono::system_clock> stime, ftime;
  chrono::duration<double> runtime;

  // run tests
  for (int k=0; k<nsizes; k++) {

    // set the problem size
    n = nvals[k];

    // allocate the matrices & vectors of this size
    A = new double*[n];
    for (i=0; i<n; i++)  A[i] = new double[n];
    R = new double*[n];
    for (i=0; i<n; i++)  R[i] = new double[n];
    Rt = new double*[n];
    for (i=0; i<n; i++)  Rt[i] = new double[n];
    xtrue = new double[n];
    x = new double[n];
    y = new double[n];
    b = new double[n];

    // fill A and xtrue with values
    for (i=0; i<n; i++)
      for (j=0; j<n; j++)
	A[i][j] = 1.0/(1.0 + 5.0*fabs(i - j));
    for (i=0; i<n; i++)
      xtrue[i] = 1.0*(1 - i)/n;


    // start first test
    cout << "\nTesting Cholesky (IP) factorizations: n = " << n << "\n";

    // compute b from A and xtrue, copy A into R
    matvec_row(A, xtrue, b, n, n);
    for (i=0; i<n; i++)
      for (j=0; j<n; j++)
	R[i][j] = A[i][j];

    // compute Cholesky decomposition
    stime = chrono::system_clock::now();
    if (cholesky_ip(R, n))
      cerr << "Error in Cholesky (IP) decomposition\n";
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   cholesky time = %.4f\n", runtime.count());

    // fill Rt as the transpose of R
    for (i=0; i<n; i++)
      for (j=0; j<n; j++)
	Rt[i][j] = R[j][i];

    // solve linear system
    stime = chrono::system_clock::now();
    if (fwdsub_row(Rt, y, b, n))
      cerr << "Error in forward substitution\n";
    if (bwdsub_row(R, x, y, n))
      cerr << "Error in backward substitution\n";
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   solve time = %.4f\n", runtime.count());

    // check error
    err_norm = 0.0;
    for (i=0; i<n; i++)
      err_norm += (x[i]-xtrue[i])*(x[i]-xtrue[i]);
    err_norm = sqrt(err_norm);
    printf("   solution error = %.4e\n", err_norm);


    // start second test
    cout << "\nTesting Cholesky (OP) factorizations: n = " << n << "\n";

    // compute b from A and xtrue, copy A into R
    matvec_row(A, xtrue, b, n, n);
    for (i=0; i<n; i++)
      for (j=0; j<n; j++)
	R[i][j] = A[i][j];

    // compute Cholesky decomposition
    stime = chrono::system_clock::now();
    if (cholesky_op(R, n))
      cerr << "Error in Cholesky (OP) decomposition\n";
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   cholesky time = %.4f\n", runtime.count());

    // fill Rt as the transpose of R
    for (i=0; i<n; i++)
      for (j=0; j<n; j++)
	Rt[i][j] = R[j][i];

    // solve linear system
    stime = chrono::system_clock::now();
    if (fwdsub_row(Rt, y, b, n))
      cerr << "Error in forward substitution\n";
    if (bwdsub_row(R, x, y, n))
      cerr << "Error in backward substitution\n";
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   solve time = %.4f\n", runtime.count());

    // check error
    err_norm = 0.0;
    for (i=0; i<n; i++)
      err_norm += (x[i]-xtrue[i])*(x[i]-xtrue[i]);
    err_norm = sqrt(err_norm);
    printf("   solution error = %.4e\n", err_norm);


    // clean up for next test
    for (i=0; i<n; i++) delete[] A[i];
    delete[] A;
    for (i=0; i<n; i++) delete[] R[i];
    delete[] R;
    for (i=0; i<n; i++) delete[] Rt[i];
    delete[] Rt;
    delete[] x;
    delete[] y;
    delete[] b;

  }

} // end main



// row-based matrix-vector product
//    A is a matrix (n x m) -- not modified
//    x is a vector (m) -- not modified
//    b is a vector (n) -- modified [must be a different pointer than x]
void matvec_row(double **A, double *x, double *b, int n, int m)
{
  // local variables
  int i, j;

  // initialize output
  for (i=0; i<n; i++)  b[i] = 0.0;

  // perform product
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      b[i] += A[i][j]*x[j];
}



// Row-oriented forward substitution
//    L is assumed lower triangular (n x n) -- not modified
//    b is the right-hand side (n) -- not modified
//    y is the solution on return (n) -- modified [may be the same pointer as b]
// returns 0 (success) or 1 (failure)
int fwdsub_row(double **L, double *y, double *b, int n)
{
  // local variables
  int i, j;

  // check nonsingular
  for (i=0; i<n; i++)
    if (L[i][i] == 0.0)  return 1;

  // loop over matrix rows, performing solve
  for (i=0; i<n; i++) {
    y[i] = b[i];           // initialize result
    for (j=0; j<i; j++)    // update this rhs
      y[i] -= L[i][j]*y[j];
    y[i] /= L[i][i];       // solve row
  }

  // return success
  return 0;
}



// Row-oriented backward substitution
//    U is assumed upper triangular (n x n) -- not modified
//    y is the right-hand side (n) -- not modified
//    x is the solution on return (n) -- modified [may be the same pointer as y]
// returns 0 (success) or 1 (failure)
int bwdsub_row(double **U, double *x, double *y, int n)
{
  // local variables
  int i, j;

  // check nonsingular
  for (i=0; i<n; i++)
    if (U[i][i] == 0.0)  return 1;

  // loop over matrix rows, performing solve
  for (i=n-1; i>=0; i--) {
    x[i] = y[i];           // initialize result
    for (j=i+1; j<n; j++)    // update this rhs
      x[i] -= U[i][j]*x[j];
    x[i] /= U[i][i];       // solve row
  }

  // return success
  return 0;
}



// Cholesky factorization (inner product version)
//    A is a symmetric matrix (n x n) -- modified
// returns 0 (success) or 1 (failure)
int cholesky_ip(double **A, int n)
{
  // local variables
  int i, j, k;

  // perform factorization in-place
  for (i=0; i<n; i++) {       // loop over rows of result
    for (k=0; k<i; k++)       // update diagonal
      A[i][i] -= A[k][i]*A[k][i];
    if (A[i][i] <= 0.0)       // check positive definite
      return 1;
    A[i][i] = sqrt(A[i][i]);  // set diagonal entry for row
    for (j=i+1; j<n; j++) {   // loop over remainder of row
      for (k=0; k<i; k++)     // update row entry
	A[i][j] -= A[k][i]*A[k][j];
      A[i][j] /= A[i][i];     // set row entry
    }
  }

  // return success
  return 0;
}



// Cholesky factorization (outer product version)
//    A is a symmetric matrix (n x n) -- modified
// returns 0 (success) or 1 (failure)
int cholesky_op(double **A, int n)
{
  // local variables
  int i, j, k;

  // perform factorization in-place
  for (i=0; i<n; i++) {       // loop over rows of result
    if (A[i][i] <= 0.0)       // check positive definite
      return 1;
    A[i][i] = sqrt(A[i][i]);  // set diagonal entry for row
    for (j=i+1; j<n; j++)     // update row
      A[i][j] /= A[i][i];
    for (k=i+1; k<n; k++)     // update remainder of matrix
      for (j=i+1; j<n; j++)
	A[k][j] -= A[i][k]*A[i][j];
  }

  // return success
  return 0;
}

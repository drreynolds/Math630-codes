/* Daniel R. Reynolds
   Math 630 @ UMBC
   Spring 2026 */

// Inclusions
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

// function prototypes
void matmat_ijk(double **A, double **X, double **B, int n, int m, int p);
void matmat_ikj(double **A, double **X, double **B, int n, int m, int p);
void matmat_jik(double **A, double **X, double **B, int n, int m, int p);
void matmat_jki(double **A, double **X, double **B, int n, int m, int p);
void matmat_kij(double **A, double **X, double **B, int n, int m, int p);
void matmat_kji(double **A, double **X, double **B, int n, int m, int p);


// main driver routine
int main(int argc, char* argv[]) {

  // local variables
  int n, m, p, i, j;
  double **A, **X, **B;
  int mvals[] = {50, 100, 200, 400};
  int nvals[] = {100, 200, 400, 800};
  int pvals[] = {75, 150, 300, 600};
  int nsizes = 4;
  chrono::time_point<chrono::system_clock> stime, ftime;
  chrono::duration<double> runtime;

  // run tests
  for (int k=0; k<nsizes; k++) {

    // set the problem size
    m = mvals[k];
    n = nvals[k];
    p = pvals[k];

    // display current problem size
    cout << "\nTesting matrix-matrix products: m = " << m << " n = " << n << " p = " << p << "\n";

    // allocate the matrix & vectors of this size
    A = new double*[m];
    for (i=0; i<m; i++)  A[i] = new double[n];
    X = new double*[n];
    for (i=0; i<n; i++)  X[i] = new double[p];
    B = new double*[m];
    for (i=0; i<m; i++)  B[i] = new double[p];

    // fill A and X with values
    for (i=0; i<m; i++)
      for (j=0; j<n; j++)
	A[i][j] = 1.0*(1 + i - j)/(n+m);
    for (i=0; i<n; i++)
      for (j=0; j<p; j++)
	X[i][j] = 1.0*(1 - i + j)/(n+p);

    // perform product 1
    stime = chrono::system_clock::now();
    matmat_ijk(A, X, B, m, n, p);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matmat_ijk:  time = %.4f\n", runtime.count());

    // perform product 2
    stime = chrono::system_clock::now();
    matmat_ikj(A, X, B, m, n, p);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matmat_ikj:  time = %.4f\n", runtime.count());

    // perform product 3
    stime = chrono::system_clock::now();
    matmat_jik(A, X, B, m, n, p);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matmat_jik:  time = %.4f\n", runtime.count());

    // perform product 4
    stime = chrono::system_clock::now();
    matmat_jki(A, X, B, m, n, p);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matmat_jki:  time = %.4f\n", runtime.count());

    // perform product 5
    stime = chrono::system_clock::now();
    matmat_kij(A, X, B, m, n, p);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matmat_kij:  time = %.4f\n", runtime.count());

    // perform product 6
    stime = chrono::system_clock::now();
    matmat_kji(A, X, B, m, n, p);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matmat_kji:  time = %.4f\n", runtime.count());

    // clean up for next test
    for (i=0; i<m; i++) delete[] A[i];
    delete[] A;
    for (i=0; i<n; i++) delete[] X[i];
    delete[] X;
    for (i=0; i<m; i++) delete[] B[i];
    delete[] B;

  }

} // end main



// matrix-matrix product (version 1)
//    A is a matrix (m x n)
//    X is a vector (n x p)
//    B is a vector (m x p)
void matmat_ijk(double **A, double **X, double **B, int m, int n, int p)
{
  // local variables
  int i, j, k;

  // initialize output
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      B[i][j] = 0.0;

  // perform product
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      for (k=0; k<n; k++)
	B[i][j] += A[i][k]*X[k][j];
}


// matrix-matrix product (version 2)
//    A is a matrix (m x n)
//    X is a vector (n x p)
//    B is a vector (m x p)
void matmat_ikj(double **A, double **X, double **B, int m, int n, int p)
{
  // local variables
  int i, j, k;

  // initialize output
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      B[i][j] = 0.0;

  // perform product
  for (i=0; i<m; i++)
    for (k=0; k<n; k++)
      for (j=0; j<p; j++)
	B[i][j] += A[i][k]*X[k][j];
}


// matrix-matrix product (version 3)
//    A is a matrix (m x n)
//    X is a vector (n x p)
//    B is a vector (m x p)
void matmat_jik(double **A, double **X, double **B, int m, int n, int p)
{
  // local variables
  int i, j, k;

  // initialize output
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      B[i][j] = 0.0;

  // perform product
  for (j=0; j<p; j++)
    for (i=0; i<m; i++)
      for (k=0; k<n; k++)
	B[i][j] += A[i][k]*X[k][j];
}


// matrix-matrix product (version 4)
//    A is a matrix (m x n)
//    X is a vector (n x p)
//    B is a vector (m x p)
void matmat_jki(double **A, double **X, double **B, int m, int n, int p)
{
  // local variables
  int i, j, k;

  // initialize output
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      B[i][j] = 0.0;

  // perform product
  for (j=0; j<p; j++)
    for (k=0; k<n; k++)
      for (i=0; i<m; i++)
	B[i][j] += A[i][k]*X[k][j];
}


// matrix-matrix product (version 5)
//    A is a matrix (m x n)
//    X is a vector (n x p)
//    B is a vector (m x p)
void matmat_kij(double **A, double **X, double **B, int m, int n, int p)
{
  // local variables
  int i, j, k;

  // initialize output
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      B[i][j] = 0.0;

  // perform product
  for (k=0; k<n; k++)
    for (i=0; i<m; i++)
      for (j=0; j<p; j++)
	B[i][j] += A[i][k]*X[k][j];
}


// matrix-matrix product (version 6)
//    A is a matrix (m x n)
//    X is a vector (n x p)
//    B is a vector (m x p)
void matmat_kji(double **A, double **X, double **B, int m, int n, int p)
{
  // local variables
  int i, j, k;

  // initialize output
  for (i=0; i<m; i++)
    for (j=0; j<p; j++)
      B[i][j] = 0.0;

  // perform product
  for (k=0; k<n; k++)
    for (j=0; j<p; j++)
      for (i=0; i<m; i++)
	B[i][j] += A[i][k]*X[k][j];
}

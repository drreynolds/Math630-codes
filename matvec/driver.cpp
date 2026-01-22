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
void matvec_row(double **A, double *x, double *b, int n, int m);
void matvec_col(double **A, double *x, double *b, int n, int m);


// main driver routine
int main(int argc, char* argv[]) {

  // local variables
  int n, m, i, j;
  double **A, *x, *b;
  int mvals[] = {1000, 2000, 4000, 8000};
  int nvals[] = {2000, 4000, 8000, 16000};
  int nsizes = 4;
  chrono::time_point<chrono::system_clock> stime, ftime;
  chrono::duration<double> runtime;

  // run tests
  for (int k=0; k<nsizes; k++) {

    // set the problem size
    m = mvals[k];
    n = nvals[k];

    // display current problem size
    cout << "\nTesting matrix-vector products with a " << m << " x " << n << " matrix:\n";

    // allocate the matrix & vectors of this size
    A = new double*[m];
    for (i=0; i<m; i++)  A[i] = new double[n];
    x = new double[n];
    b = new double[m];

    // fill A and x with values
    for (i=0; i<m; i++)
      for (j=0; j<n; j++)
	A[i][j] = 1.0*(1 + i - j)/(n+m);
    for (j=0; j<n; j++)
      x[j] = 1.0*j/n;

    // perform row-based product
    stime = chrono::system_clock::now();
    matvec_row(A, x, b, m, n);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matvec_row:  time = %.4f\n", runtime.count());

    // perform column-based product
    stime = chrono::system_clock::now();
    matvec_col(A, x, b, m, n);
    ftime = chrono::system_clock::now();
    runtime = ftime-stime;
    printf("   matvec_col:  time = %.4f\n", runtime.count());

    // clean up for next test
    for (i=0; i<m; i++) delete[] A[i];
    delete[] A;
    delete[] x;
    delete[] b;

  }

} // end main



// row-based matrix-vector product
//    A is a matrix (m x n)
//    x is a vector (n)
//    b is a vector (m)
void matvec_row(double **A, double *x, double *b, int m, int n)
{
  // local variables
  int i, j;

  // initialize output
  for (i=0; i<m; i++)  b[i] = 0.0;

  // perform product
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      b[i] += A[i][j]*x[j];
}



// column-based matrix-vector product
//    A is a matrix (m x n)
//    x is a vector (n)
//    b is a vector (m)
void matvec_col(double **A, double *x, double *b, int m, int n)
{
  // local variables
  int i, j;

  // initialize output
  for (i=0; i<m; i++)  b[i] = 0.0;

  // perform product
  for (j=0; j<n; j++)
    for (i=0; i<m; i++)
      b[i] += A[i][j]*x[j];
}

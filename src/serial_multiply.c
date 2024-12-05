/* src/serial_multiply.c */
#include "matrix_mult.h"
#include <omp.h>

/* Serial matrix multiplication using ijk order */
void serial_multiply_ijk(double **a, double **b, double **c, int N)
{
    int i, j, k;

    /* Initialize matrix c to zero.
       - `#pragma omp parallel for`: Allows parallel execution of the loop.
       - `private(i)`: Ensures each thread has its own copy of variable i.
       - `num_threads(1)`: Forces the loop to run serially (only one thread). */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in ijk order:
           - Outer loop iterates over rows of matrix a.
           - Middle loop iterates over columns of matrix b.
           - Inner loop performs the dot product. */
#pragma omp parallel for private(i, j, k) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
}

/* Serial matrix multiplication using ikj order */
void serial_multiply_ikj(double **a, double **b, double **c, int N)
{
    int i, j, k;
    double temp;

    /* Initialize matrix c to zero. */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in ikj order:
           - Outer loop iterates over rows of matrix a.
           - Middle loop iterates over rows of matrix b.
           - Inner loop performs the dot product using a temporary variable `temp`
             to optimize cache usage. */
#pragma omp parallel for private(i, k, j, temp) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
    {
        for (k = 0; k < N; k++)
        {
            temp = a[i][k];
            for (j = 0; j < N; j++)
                c[i][j] += temp * b[k][j];
        }
    }
}

/* Serial matrix multiplication using jik order */
void serial_multiply_jik(double **a, double **b, double **c, int N)
{
    int i, j, k;

    /* Initialize matrix c to zero. */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in jik order:
           - Outer loop iterates over columns of matrix b.
           - Middle loop iterates over rows of matrix a.
           - Inner loop performs the dot product. */
#pragma omp parallel for private(j, i, k) schedule(static) num_threads(1)
    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
}

/* Serial matrix multiplication using jki order */
void serial_multiply_jki(double **a, double **b, double **c, int N)
{
    int i, j, k;
    double temp;

    /* Initialize matrix c to zero. */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in jki order:
           - Outer loop iterates over columns of matrix b.
           - Middle loop iterates over rows of matrix b.
           - Inner loop updates rows of matrix a using `temp`. */
#pragma omp parallel for private(j, k, i, temp) schedule(static) num_threads(1)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
}

/* Serial matrix multiplication using kij order */
void serial_multiply_kij(double **a, double **b, double **c, int N)
{
    int i, j, k;
    double temp;

    /* Initialize matrix c to zero. */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in kij order:
           - Outer loop iterates over rows of matrix b.
           - Middle loop updates rows of matrix a.
           - Inner loop iterates over columns of matrix b using `temp`. */
#pragma omp parallel for private(k, i, j, temp) schedule(static) num_threads(1)
    for (k = 0; k < N; k++)
        for (i = 0; i < N; i++)
        {
            temp = a[i][k];
            for (j = 0; j < N; j++)
                c[i][j] += temp * b[k][j];
        }
}

/* Serial matrix multiplication using kji order */
void serial_multiply_kji(double **a, double **b, double **c, int N)
{
    int i, j, k;
    double temp;

    /* Initialize matrix c to zero. */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in kji order:
           - Outer loop iterates over rows of matrix b.
           - Middle loop updates rows of matrix a.
           - Inner loop iterates over columns of matrix b using `temp`. */
#pragma omp parallel for private(k, j, i, temp) schedule(static) num_threads(1)
    for (k = 0; k < N; k++)
        for (j = 0; j < N; j++)
        {
            temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
}

/* Serial blocked matrix multiplication:
   Divides matrices into blocks of size `block_size` and performs
   multiplication block by block. This improves cache efficiency. */
void serial_blocked_multiply(double **a, double **b, double **c, int N, int block_size)
{
    int i0, j0, k0, i, j, k;
    double temp;

    /* Initialize matrix c to zero. */
#pragma omp parallel for private(i) schedule(static) num_threads(1)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform blocked matrix multiplication:
           - Loops iterate over blocks of size `block_size`.
           - Nested loops perform multiplication within each block. */
#pragma omp parallel for collapse(2) private(i0, j0, k0, i, j, k, temp) schedule(static) num_threads(1)
    for (i0 = 0; i0 < N; i0 += block_size)
        for (k0 = 0; k0 += block_size)
            for (j0 = 0; j0 < N; j0 += block_size)
                for (i = i0; i < ((i0 + block_size) > N ? N : (i0 + block_size)); i++)
                    for (k = k0; k < ((k0 + block_size) > N ? N : (k0 + block_size)); k++)
                    {
                        temp = a[i][k];
                        for (j = j0; j < ((j0 + block_size) > N ? N : (j0 + block_size)); j++)
                            c[i][j] += temp * b[k][j];
                    }
}

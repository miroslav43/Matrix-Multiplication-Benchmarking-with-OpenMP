/* src/parallel_multiply.c */
#include "matrix_mult.h"

/* Parallel matrix multiplication using ijk order */
void parallel_multiply_ijk(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;

    /* Initialize matrix c to zero using parallel processing */
#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in ijk order:
           - Outer loop iterates over rows of matrix a.
           - Middle loop iterates over columns of matrix b.
           - Inner loop performs the dot product.
           - OpenMP parallelism is used to distribute iterations of the outermost loop (i)
             across threads, improving performance.
           - `schedule(static, chunk)` specifies that iterations are divided into fixed-size chunks
             (defined by `chunk`) and assigned to threads. */
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
}

/* Parallel matrix multiplication using ikj order */
void parallel_multiply_ikj(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;
    double temp;

    /* Initialize matrix c to zero using parallel processing */
#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in ikj order:
           - Outer loop iterates over rows of matrix a.
           - Middle loop iterates over rows of matrix b.
           - Inner loop performs the dot product.
           - `temp` is used to store a[i][k] temporarily to optimize cache usage. */
#pragma omp parallel for private(i, j, k, temp) schedule(static, chunk) num_threads(nthreads)
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

/* Parallel matrix multiplication using jik order */
void parallel_multiply_jik(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;

    /* Initialize matrix c to zero using parallel processing */
#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in jik order:
           - Outer loop iterates over columns of matrix b.
           - Middle loop iterates over rows of matrix a.
           - Inner loop performs the dot product. */
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
}

/* Parallel matrix multiplication using jki order */
void parallel_multiply_jki(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;
    double temp;

    /* Initialize matrix c to zero using parallel processing */
#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

        /* Perform matrix multiplication in jki order:
           - Outer loop iterates over columns of matrix b.
           - Middle loop iterates over rows of matrix b.
           - Inner loop updates rows of matrix a. */
#pragma omp parallel for private(i, j, k, temp) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
}

/* Parallel matrix multiplication using kij order */
void parallel_multiply_kij(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;
    double temp;

    /* Perform matrix multiplication in kij order:
       - Outer loop iterates over rows of matrix b.
       - Middle loop updates rows of matrix a.
       - Inner loop iterates over columns of matrix b. */
#pragma omp parallel for private(i, j, k, temp) schedule(static, chunk) num_threads(nthreads)
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

/* Parallel matrix multiplication using kji order */
void parallel_multiply_kji(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int j, k, i;
    double temp;

    /* Perform matrix multiplication in kji order:
       - Outer loop iterates over rows of matrix b.
       - Middle loop updates rows of matrix a.
       - Inner loop iterates over columns of matrix b. */
#pragma omp parallel for private(i, j, k, temp) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
    {
        for (k = 0; k < N; k++)
        {
            temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
    }
}

/* Parallel blocked matrix multiplication */
void parallel_blocked_multiply(double **a, double **b, double **c, int N, int block_size, int nthreads)
{
    int i0, j0, k0, i, j, k;
    double temp;

    /* Perform blocked matrix multiplication:
       - Divide matrices into blocks of size `block_size`.
       - Process each block independently using nested loops.
       - Collapsing two outermost loops allows OpenMP to parallelize
         the blocks efficiently.
       - Schedule blocks statically for better load balancing. */
#pragma omp parallel for collapse(2) private(i0, j0, k0, i, j, k, temp) schedule(static) num_threads(nthreads)
    for (i0 = 0; i0 < N; i0 += block_size)
        for (j0 = 0; j0 < N; j0 += block_size)
            for (k0 = 0; k0 < N; k0 += block_size)
                for (i = i0; i < ((i0 + block_size) > N ? N : (i0 + block_size)); i++)
                    for (k = k0; k < ((k0 + block_size) > N ? N : (k0 + block_size)); k++)
                    {
                        temp = a[i][k];
                        for (j = j0; j < ((j0 + block_size) > N ? N : (j0 + block_size)); j++)
                            c[i][j] += temp * b[k][j];
                    }
}

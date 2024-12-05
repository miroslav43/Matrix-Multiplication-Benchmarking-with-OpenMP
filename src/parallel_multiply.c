/* src/parallel_multiply.c */
#include "matrix_mult.h"

void parallel_multiply_ijk(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;

#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
}

void parallel_multiply_ikj(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;
    double temp;

#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

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

void parallel_multiply_jik(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;

#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
}

void parallel_multiply_jki(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;
    double temp;

#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        memset(c[i], 0, N * sizeof(double));

#pragma omp parallel for private(i, j, k, temp) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
}

void parallel_multiply_kij(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int i, j, k;
    double temp;

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

void parallel_multiply_kji(double **a, double **b, double **c, int N, int nthreads, int chunk)
{
    int j, k, i;
    double temp;

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

void parallel_blocked_multiply(double **a, double **b, double **c, int N, int block_size, int nthreads)
{
    int i0, j0, k0, i, j, k;
    double temp;

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

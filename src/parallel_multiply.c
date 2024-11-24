/* src/parallel_multiply.c */
#include "matrix_mult.h"

void parallel_multiply_ijk(int nthreads, int chunk)
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            c2[i][j] = 0;
            for (k = 0; k < N; k++)
                c2[i][j] += a[i][k] * b[k][j];
        }
}

void parallel_multiply_ikj(int nthreads, int chunk)
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            c2[i][j] = 0;
        for (k = 0; k < N; k++)
        {
            double temp = a[i][k];
            for (j = 0; j < N; j++)
                c2[i][j] += temp * b[k][j];
        }
    }
}

void parallel_multiply_jik(int nthreads, int chunk)
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
        {
            c2[i][j] = 0;
            for (k = 0; k < N; k++)
                c2[i][j] += a[i][k] * b[k][j];
        }
}

void parallel_multiply_jki(int nthreads, int chunk)
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            double temp = b[k][j];
            for (i = 0; i < N; i++)
                c2[i][j] += a[i][k] * temp;
        }
}

void parallel_multiply_kij(int nthreads, int chunk)
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (k = 0; k < N; k++)
        for (i = 0; i < N; i++)
        {
            double temp = a[i][k];
            for (j = 0; j < N; j++)
#pragma omp atomic
                c2[i][j] += temp * b[k][j];
        }
}

void parallel_multiply_kji(int nthreads, int chunk)
{
    int i, j, k;
#pragma omp parallel for private(i, j, k) schedule(static, chunk) num_threads(nthreads)
    for (k = 0; k < N; k++)
        for (j = 0; j < N; j++)
        {
            double temp = b[k][j];
            for (i = 0; i < N; i++)
#pragma omp atomic
                c2[i][j] += a[i][k] * temp;
        }
}

void parallel_blocked_multiply(int block_size, int nthreads)
{
    int i0, j0, k0, i, j, k;

// Initialize c2 matrix to zero
#pragma omp parallel for private(i, j) schedule(static) num_threads(nthreads)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c2[i][j] = 0;

// Perform blocked matrix multiplication in parallel
#pragma omp parallel for collapse(2) private(i0, j0, k0, i, j, k) schedule(dynamic) num_threads(nthreads)
    for (i0 = 0; i0 < N; i0 += block_size)
        for (k0 = 0; k0 < N; k0 += block_size)
            for (j0 = 0; j0 < N; j0 += block_size)
                for (i = i0; i < ((i0 + block_size) < N ? (i0 + block_size) : N); i++)
                    for (k = k0; k < ((k0 + block_size) < N ? (k0 + block_size) : N); k++)
                    {
                        double temp = a[i][k];
                        for (j = j0; j < ((j0 + block_size) < N ? (j0 + block_size) : N); j++)
                            c2[i][j] += temp * b[k][j];
                    }
}

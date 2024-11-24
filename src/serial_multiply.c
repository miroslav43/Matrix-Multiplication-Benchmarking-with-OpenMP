/* src/serial_multiply.c */
#include "matrix_mult.h"

void serial_multiply_ijk()
{
    int i, j, k;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            c[i][j] = 0;
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
}

void serial_multiply_ikj()
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            c[i][j] = 0;
        for (k = 0; k < N; k++)
        {
            double temp = a[i][k];
            for (j = 0; j < N; j++)
                c[i][j] += temp * b[k][j];
        }
    }
}

void serial_multiply_jik()
{
    int i, j, k;
    for (j = 0; j < N; j++)
        for (i = 0; i < N; i++)
        {
            c[i][j] = 0;
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
}

void serial_multiply_jki()
{
    int i, j, k;
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
        {
            double temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
}

void serial_multiply_kij()
{
    int i, j, k;
    for (k = 0; k < N; k++)
        for (i = 0; i < N; i++)
        {
            double temp = a[i][k];
            for (j = 0; j < N; j++)
                c[i][j] += temp * b[k][j];
        }
}

void serial_multiply_kji()
{
    int i, j, k;
    for (k = 0; k < N; k++)
        for (j = 0; j < N; j++)
        {
            double temp = b[k][j];
            for (i = 0; i < N; i++)
                c[i][j] += a[i][k] * temp;
        }
}

void serial_blocked_multiply(int block_size)
{
    int i0, j0, k0, i, j, k;

    // Initialize c matrix to zero
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            c[i][j] = 0;

    // Perform blocked matrix multiplication
    for (i0 = 0; i0 < N; i0 += block_size)
        for (k0 = 0; k0 < N; k0 += block_size)
            for (j0 = 0; j0 < N; j0 += block_size)
                for (i = i0; i < ((i0 + block_size) < N ? (i0 + block_size) : N); i++)
                    for (k = k0; k < ((k0 + block_size) < N ? (k0 + block_size) : N); k++)
                    {
                        double temp = a[i][k];
                        for (j = j0; j < ((j0 + block_size) < N ? (j0 + block_size) : N); j++)
                            c[i][j] += temp * b[k][j];
                    }
}

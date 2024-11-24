/* src/matrix_mult.h */
#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#define EPSILON 0.000001
#define MAX_SIZE 3000  // Maximum matrix size

extern int N;  // Matrix size (N x N)
extern int BS; // Block size

extern double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE], c2[MAX_SIZE][MAX_SIZE], c_gt[MAX_SIZE][MAX_SIZE];

void generate_matrix(double mat[MAX_SIZE][MAX_SIZE]);

// Serial multiplication functions with loop orders in function names
void serial_multiply_ijk();
void serial_multiply_ikj();
void serial_multiply_jik();
void serial_multiply_jki();
void serial_multiply_kij();
void serial_multiply_kji();
void serial_blocked_multiply(int block_size);

// Parallel multiplication functions with loop orders in function names
void parallel_multiply_ijk(int nthreads, int chunk);
void parallel_multiply_ikj(int nthreads, int chunk);
void parallel_multiply_jik(int nthreads, int chunk);
void parallel_multiply_jki(int nthreads, int chunk);
void parallel_multiply_kij(int nthreads, int chunk);
void parallel_multiply_kji(int nthreads, int chunk);
void parallel_blocked_multiply(int block_size, int nthreads);

#endif

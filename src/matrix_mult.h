/* src/matrix_mult.h */
#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define EPSILON 0.000001

// Function prototypes for matrix operations
double **allocate_matrix(int N);
void free_matrix(double **mat, int N);
void generate_matrix(double **mat, int N);
void initialize_matrix(double **mat, int N);
void copy_matrix(double **dest, double **src, int N);
double compute_matrix_checksum(double **mat, int N);

// Serial multiplication functions
void serial_multiply_ijk(double **a, double **b, double **c, int N);
void serial_multiply_ikj(double **a, double **b, double **c, int N);
void serial_multiply_jik(double **a, double **b, double **c, int N);
void serial_multiply_jki(double **a, double **b, double **c, int N);
void serial_multiply_kij(double **a, double **b, double **c, int N);
void serial_multiply_kji(double **a, double **b, double **c, int N);
void serial_blocked_multiply(double **a, double **b, double **c, int N, int block_size);

// Parallel multiplication functions
void parallel_multiply_ijk(double **a, double **b, double **c, int N, int nthreads, int chunk);
void parallel_multiply_ikj(double **a, double **b, double **c, int N, int nthreads, int chunk);
void parallel_multiply_jik(double **a, double **b, double **c, int N, int nthreads, int chunk);
void parallel_multiply_jki(double **a, double **b, double **c, int N, int nthreads, int chunk);
void parallel_multiply_kij(double **a, double **b, double **c, int N, int nthreads, int chunk);
void parallel_multiply_kji(double **a, double **b, double **c, int N, int nthreads, int chunk);
void parallel_blocked_multiply(double **a, double **b, double **c, int N, int block_size, int nthreads);

#endif

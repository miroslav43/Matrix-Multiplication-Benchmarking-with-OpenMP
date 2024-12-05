#include "matrix_mult.h"
#include "validation.h"
#include <stdio.h>
#include <stdlib.h>

/* Allocates memory for a square matrix of size N x N */
double **allocate_matrix(int N)
{
    double **mat = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        mat[i] = (double *)calloc(N, sizeof(double));
    return mat;
}

/* Frees the memory allocated for a square matrix */
void free_matrix(double **mat, int N)
{
    for (int i = 0; i < N; i++)
        free(mat[i]);
    free(mat);
}

/* Populates a square matrix with random values between 1 and 10 */
void generate_matrix(double **mat, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = rand() % 10 + 1;
}

/* Initializes a square matrix to all zeros */
void initialize_matrix(double **mat, int N)
{
    for (int i = 0; i < N; i++)
        memset(mat[i], 0, N * sizeof(double));
}

/* Copies the contents of one matrix (src) into another (dest) */
void copy_matrix(double **dest, double **src, int N)
{
    for (int i = 0; i < N; i++)
        memcpy(dest[i], src[i], N * sizeof(double));
}

/* Computes the checksum (sum of all elements) of a square matrix */
double compute_matrix_checksum(double **mat, int N)
{
    double checksum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            checksum += mat[i][j];
    return checksum;
}

/* Main function for matrix multiplication performance testing */
int main(int argc, char *argv[])
{
    int chunk = 10; // Chunk size for parallel processing
    double start, end, time_serial, time_parallel;

    // Open CSV file to log results
    FILE *csv_file = fopen("matrix_mult_results_fopenmp.csv", "w");
    if (!csv_file)
    {
        perror("Error opening CSV file");
        return EXIT_FAILURE;
    }

    fprintf(csv_file, "N,Algorithm,Version,Threads,Time(s),Speedup,BlockSize,Result\n");
    fflush(csv_file);

    // Array of serial matrix multiplication functions
    void (*serial_funcs[6])(double **, double **, double **, int) = {
        serial_multiply_ijk,
        serial_multiply_ikj,
        serial_multiply_jik,
        serial_multiply_jki,
        serial_multiply_kij,
        serial_multiply_kji};

    // Array of parallel matrix multiplication functions
    void (*parallel_funcs[6])(double **, double **, double **, int, int, int) = {
        parallel_multiply_ijk,
        parallel_multiply_ikj,
        parallel_multiply_jik,
        parallel_multiply_jki,
        parallel_multiply_kij,
        parallel_multiply_kji};

    const char *func_names[6] = {
        "ijk", "ikj", "jik", "jki", "kij", "kji"};

    int N_values[] = {1000, 1500, 2000, 2500, 3000}; // Matrix sizes to test
    int thread_counts[] = {4, 6, 8, 10, 16}; // Number of threads to test

    // Loop through different matrix sizes
    for (int n_idx = 0; n_idx < sizeof(N_values) / sizeof(N_values[0]); n_idx++)
    {
        int N = N_values[n_idx];
        printf("\n=== Testing for N = %d ===\n", N);

        srand(0);

        // Allocate matrices
        double **a = allocate_matrix(N);
        double **b = allocate_matrix(N);
        double **c = allocate_matrix(N);
        double **c2 = allocate_matrix(N);
        double **c_gt = allocate_matrix(N);

        // Generate random values for matrices a and b
        generate_matrix(a, N);
        generate_matrix(b, N);

        // Compute initial checksums for verification
        double checksum_a_before = compute_matrix_checksum(a, N);
        double checksum_b_before = compute_matrix_checksum(b, N);

        // Compute ground truth result using serial ijk multiplication
        printf("Starting serial ijk multiplication (Ground Truth)...\n");
        initialize_matrix(c, N);
        start = omp_get_wtime();
        serial_multiply_ijk(a, b, c, N);
        end = omp_get_wtime();
        double time_ground_truth = end - start;

        double checksum_c_gt = compute_matrix_checksum(c, N);

        // Save ground truth result
        copy_matrix(c_gt, c, N);

        fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, "ijk", "Serial", 1, time_ground_truth, 1.0, 0, "Valid");
        fflush(csv_file);

        // Test other serial algorithms
        for (int idx = 0; idx < 6; idx++)
        {
            initialize_matrix(c, N);

            printf("\nStarting serial %s multiplication...\n", func_names[idx]);
            start = omp_get_wtime();
            serial_funcs[idx](a, b, c, N);
            end = omp_get_wtime();
            time_serial = end - start;

            double checksum_c = compute_matrix_checksum(c, N);
            int valid = validate_results(func_names[idx], c, c_gt, N);

            fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, func_names[idx], "Serial", 1, time_serial, time_ground_truth / time_serial, 0, valid ? "Valid" : "Mismatch");
            fflush(csv_file);

            // Test parallel versions
            for (int t_idx = 0; t_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); t_idx++)
            {
                int nthreads = thread_counts[t_idx];

                initialize_matrix(c2, N);

                printf("\nStarting parallel %s multiplication with %d threads...\n", func_names[idx], nthreads);

                start = omp_get_wtime();
                parallel_funcs[idx](a, b, c2, N, nthreads, chunk);
                end = omp_get_wtime();
                time_parallel = end - start;

                double checksum_c2 = compute_matrix_checksum(c2, N);
                int valid = validate_results(func_names[idx], c2, c_gt, N);

                fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, func_names[idx], "Parallel", nthreads, time_parallel, time_serial / time_parallel, 0, valid ? "Valid" : "Mismatch");
                fflush(csv_file);
            }
        }

        // Test blocked algorithms
        int block_sizes[] = {16, 32, 64, 128, 256};
        for (int bs_idx = 0; bs_idx < sizeof(block_sizes) / sizeof(block_sizes[0]); bs_idx++)
        {
            int BS = block_sizes[bs_idx];

            initialize_matrix(c, N);

            printf("Starting serial blocked multiplication...\n");
            start = omp_get_wtime();
            serial_blocked_multiply(a, b, c, N, BS);
            end = omp_get_wtime();
            time_serial = end - start;

            double checksum_c = compute_matrix_checksum(c, N);
            int valid = validate_results("Blocked Serial", c, c_gt, N);

            fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, "Blocked", "Serial", 1, time_serial, time_ground_truth / time_serial, BS, valid ? "Valid" : "Mismatch");
            fflush(csv_file);

            for (int t_idx = 0; t_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); t_idx++)
            {
                int nthreads = thread_counts[t_idx];

                initialize_matrix(c2, N);

                start = omp_get_wtime();
                parallel_blocked_multiply(a, b, c2, N, BS, nthreads);
                end = omp_get_wtime();
                time_parallel = end - start;

                double checksum_c2 = compute_matrix_checksum(c2, N);
                int valid = validate_results("Blocked Parallel", c2, c_gt, N);

                fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, "Blocked", "Parallel", nthreads, time_parallel, time_serial / time_parallel, BS, valid ? "Valid" : "Mismatch");
                fflush(csv_file);
            }
        }

        // Free matrices
        free_matrix(a, N);
        free_matrix(b, N);
        free_matrix(c, N);
        free_matrix(c2, N);
        free_matrix(c_gt, N);
    }

    fclose(csv_file);
    printf("\nPerformance data written to matrix_mult_results_fopenmp.csv\n");
    return 0;
}

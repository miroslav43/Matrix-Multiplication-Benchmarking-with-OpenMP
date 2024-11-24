/* src/main.c */
#include "matrix_mult.h"
#include "validation.h"
#include <stdio.h>
#include <stdlib.h>

double **allocate_matrix(int N)
{
    double **mat = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        mat[i] = (double *)calloc(N, sizeof(double));
    return mat;
}

void free_matrix(double **mat, int N)
{
    for (int i = 0; i < N; i++)
        free(mat[i]);
    free(mat);
}

void generate_matrix(double **mat, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = rand() % 10 + 1; // Values between 1 and 10
}

void initialize_matrix(double **mat, int N)
{
    for (int i = 0; i < N; i++)
        memset(mat[i], 0, N * sizeof(double));
}

void copy_matrix(double **dest, double **src, int N)
{
    for (int i = 0; i < N; i++)
        memcpy(dest[i], src[i], N * sizeof(double));
}

double compute_matrix_checksum(double **mat, int N)
{
    double checksum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            checksum += mat[i][j];
    return checksum;
}

int main(int argc, char *argv[])
{
    int chunk = 10;
    double start, end, time_serial, time_parallel;

    // Open CSV file for writing
    FILE *csv_file = fopen("matrix_mult_results.csv", "w");
    if (!csv_file)
    {
        perror("Error opening CSV file");
        return EXIT_FAILURE;
    }

    // Write CSV headers
    fprintf(csv_file, "N,Algorithm,Version,Threads,Time(s),Speedup,BlockSize,Result\n");
    fflush(csv_file); // Force write to CSV

    // Array to store function pointers and names
    void (*serial_funcs[6])(double **, double **, double **, int) = {
        serial_multiply_ijk,
        serial_multiply_ikj,
        serial_multiply_jik,
        serial_multiply_jki,
        serial_multiply_kij,
        serial_multiply_kji};

    void (*parallel_funcs[6])(double **, double **, double **, int, int, int) = {
        parallel_multiply_ijk,
        parallel_multiply_ikj,
        parallel_multiply_jik,
        parallel_multiply_jki,
        parallel_multiply_kij,
        parallel_multiply_kji};

    const char *func_names[6] = {
        "ijk", "ikj", "jik", "jki", "kij", "kji"};

    int N_values[] = {100, 150, 200, 250, 300};
    // int N_values[] = {1000, 1500, 2000, 2500, 3000}; // Adjust as needed
    int thread_counts[] = {4, 6, 8}; // Testing with 4, 6, and 8 threads

    for (int n_idx = 0; n_idx < sizeof(N_values) / sizeof(N_values[0]); n_idx++)
    {
        int N = N_values[n_idx];
        printf("\n=== Testing for N = %d ===\n", N);

        // Seed the random number generator
        srand(0); // Use a fixed seed for reproducibility

        // Dynamically allocate matrices
        double **a = allocate_matrix(N);
        double **b = allocate_matrix(N);
        double **c = allocate_matrix(N);
        double **c2 = allocate_matrix(N);
        double **c_gt = allocate_matrix(N);

        // Generate matrices
        generate_matrix(a, N);
        generate_matrix(b, N);

        // Compute and print checksums of a and b
        double checksum_a_before = compute_matrix_checksum(a, N);
        double checksum_b_before = compute_matrix_checksum(b, N);
        printf("Checksum of matrix a before computation: %lf\n", checksum_a_before);
        printf("Checksum of matrix b before computation: %lf\n", checksum_b_before);

        // Ground truth using serial ijk version
        printf("Starting serial ijk multiplication (Ground Truth)...\n");
        initialize_matrix(c, N);
        start = omp_get_wtime();
        serial_multiply_ijk(a, b, c, N);
        end = omp_get_wtime();
        double time_ground_truth = end - start;
        printf("Serial ijk time: %lf seconds\n", time_ground_truth);

        double checksum_c_gt = compute_matrix_checksum(c, N);
        printf("Checksum of ground truth result c_gt: %lf\n", checksum_c_gt);

        // Copy c to c_gt
        copy_matrix(c_gt, c, N);

        // Verify that a and b have not changed
        double checksum_a_after_gt = compute_matrix_checksum(a, N);
        double checksum_b_after_gt = compute_matrix_checksum(b, N);
        if (checksum_a_before != checksum_a_after_gt || checksum_b_before != checksum_b_after_gt)
        {
            printf("Error: Matrices a or b have been modified during ground truth computation.\n");
            exit(EXIT_FAILURE);
        }

        // Write ground truth time to CSV
        fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, "ijk", "Serial", 1, time_ground_truth, 1.0, 0, "Valid");
        fflush(csv_file); // Force write to CSV

        // Loop over all permutations
        for (int idx = 0; idx < 6; idx++)
        {
            // Reset c
            initialize_matrix(c, N);

            // Serial version
            printf("\nStarting serial %s multiplication...\n", func_names[idx]);
            start = omp_get_wtime();
            serial_funcs[idx](a, b, c, N);
            end = omp_get_wtime();
            time_serial = end - start;
            printf("Serial %s time: %lf seconds\n", func_names[idx], time_serial);

            double checksum_c = compute_matrix_checksum(c, N);
            printf("Checksum of result c: %lf\n", checksum_c);

            // Validate against ground truth
            int valid = validate_results(func_names[idx], c, c_gt, N);

            // Verify that a and b have not changed
            double checksum_a_after = compute_matrix_checksum(a, N);
            double checksum_b_after = compute_matrix_checksum(b, N);
            if (checksum_a_before != checksum_a_after || checksum_b_before != checksum_b_after)
            {
                printf("Error: Matrices a or b have been modified during serial %s multiplication.\n", func_names[idx]);
                exit(EXIT_FAILURE);
            }

            // Write serial results to CSV
            fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, func_names[idx], "Serial", 1, time_serial, time_ground_truth / time_serial, 0, valid ? "Valid" : "Mismatch");
            fflush(csv_file); // Force write to CSV

            // Test with different thread counts
            for (int t_idx = 0; t_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); t_idx++)
            {
                int nthreads = thread_counts[t_idx];
                printf("\nStarting parallel %s multiplication with %d threads...\n", func_names[idx], nthreads);

                // Reset c2
                initialize_matrix(c2, N);

                start = omp_get_wtime();
                parallel_funcs[idx](a, b, c2, N, nthreads, chunk);
                end = omp_get_wtime();
                time_parallel = end - start;
                printf("Parallel %s time with %d threads: %lf seconds\n", func_names[idx], nthreads, time_parallel);
                printf("Speedup: %2.2lf\n", time_serial / time_parallel);

                double checksum_c2 = compute_matrix_checksum(c2, N);
                printf("Checksum of result c2: %lf\n", checksum_c2);

                // Validate against ground truth
                int valid = validate_results(func_names[idx], c2, c_gt, N);

                // Verify that a and b have not changed
                double checksum_a_after_parallel = compute_matrix_checksum(a, N);
                double checksum_b_after_parallel = compute_matrix_checksum(b, N);
                if (checksum_a_before != checksum_a_after_parallel || checksum_b_before != checksum_b_after_parallel)
                {
                    printf("Error: Matrices a or b have been modified during parallel %s multiplication.\n", func_names[idx]);
                    exit(EXIT_FAILURE);
                }

                // Write parallel results to CSV
                fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, func_names[idx], "Parallel", nthreads, time_parallel, time_serial / time_parallel, 0, valid ? "Valid" : "Mismatch");
                fflush(csv_file); // Force write to CSV
            }
        }

        // Blocked algorithms
        int block_sizes[] = {16, 32, 64, 128, 256};
        for (int bs_idx = 0; bs_idx < sizeof(block_sizes) / sizeof(block_sizes[0]); bs_idx++)
        {
            int BS = block_sizes[bs_idx];
            printf("\nTesting block size: %d\n", BS);

            // Reset c
            initialize_matrix(c, N);

            // Serial blocked multiplication
            printf("Starting serial blocked multiplication...\n");
            start = omp_get_wtime();
            serial_blocked_multiply(a, b, c, N, BS);
            end = omp_get_wtime();
            time_serial = end - start;
            printf("Serial blocked time: %lf seconds\n", time_serial);

            double checksum_c = compute_matrix_checksum(c, N);
            printf("Checksum of result c: %lf\n", checksum_c);

            // Validate against ground truth
            int valid = validate_results("Blocked Serial", c, c_gt, N);

            // Write serial blocked results to CSV
            fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, "Blocked", "Serial", 1, time_serial, time_ground_truth / time_serial, BS, valid ? "Valid" : "Mismatch");
            fflush(csv_file); // Force write to CSV

            // Test with different thread counts
            for (int t_idx = 0; t_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); t_idx++)
            {
                int nthreads = thread_counts[t_idx];
                printf("\nStarting parallel blocked multiplication with %d threads...\n", nthreads);

                // Reset c2
                initialize_matrix(c2, N);

                start = omp_get_wtime();
                parallel_blocked_multiply(a, b, c2, N, BS, nthreads);
                end = omp_get_wtime();
                time_parallel = end - start;
                printf("Parallel blocked time with %d threads: %lf seconds\n", nthreads, time_parallel);
                printf("Speedup: %2.2lf\n", time_serial / time_parallel);

                double checksum_c2 = compute_matrix_checksum(c2, N);
                printf("Checksum of result c2: %lf\n", checksum_c2);

                // Validate against ground truth
                int valid = validate_results("Blocked Parallel", c2, c_gt, N);

                // Write parallel blocked results to CSV
                fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d,%s\n", N, "Blocked", "Parallel", nthreads, time_parallel, time_serial / time_parallel, BS, valid ? "Valid" : "Mismatch");
                fflush(csv_file); // Force write to CSV
            }
        }

        // Free allocated matrices
        free_matrix(a, N);
        free_matrix(b, N);
        free_matrix(c, N);
        free_matrix(c2, N);
        free_matrix(c_gt, N);
    }

    fclose(csv_file);
    printf("\nPerformance data written to matrix_mult_results.csv\n");
    return 0;
}

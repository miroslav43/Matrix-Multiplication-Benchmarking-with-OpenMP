/* src/main.c */
#include "matrix_mult.h"
#include "validation.h"
#include <string.h> // For memset and memcpy
#include <stdio.h>
#include <stdlib.h>

int N;       // Matrix size will be set in the loop
int BS = 64; // Default block size

double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE], c2[MAX_SIZE][MAX_SIZE], c_gt[MAX_SIZE][MAX_SIZE];

void generate_matrix(double mat[MAX_SIZE][MAX_SIZE])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = rand() % 10 + 1; // Values between 1 and 10
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
    fprintf(csv_file, "N,Algorithm,Version,Threads,Time(s),Speedup,BlockSize\n");
    fflush(csv_file); // Force write to CSV

    // Array to store function pointers and names
    void (*serial_funcs[6])() = {
        serial_multiply_ijk,
        serial_multiply_ikj,
        serial_multiply_jik,
        serial_multiply_jki,
        serial_multiply_kij,
        serial_multiply_kji};

    void (*parallel_funcs[6])(int, int) = {
        parallel_multiply_ijk,
        parallel_multiply_ikj,
        parallel_multiply_jik,
        parallel_multiply_jki,
        parallel_multiply_kij,
        parallel_multiply_kji};

    const char *func_names[6] = {
        "ijk", "ikj", "jik", "jki", "kij", "kji"};

    int N_values[] = {1000, 1500, 2000, 2500, 3000};
    int thread_counts[] = {4, 6, 8}; // Testing with 4, 6, and 8 threads

    for (int n_idx = 0; n_idx < sizeof(N_values) / sizeof(N_values[0]); n_idx++)
    {
        N = N_values[n_idx];
        printf("\n=== Testing for N = %d ===\n", N);

        // Seed the random number generator
        srand(time(NULL));

        // Generate matrices
        generate_matrix(a);
        generate_matrix(b);

        // Ground truth using serial ijk version
        printf("Starting serial ijk multiplication (Ground Truth)...\n");
        memset(c, 0, sizeof(c));
        start = omp_get_wtime();
        serial_multiply_ijk();
        end = omp_get_wtime();
        double time_ground_truth = end - start;
        printf("Serial ijk time: %lf seconds\n", time_ground_truth);

        // Copy c to c_gt
        memcpy(c_gt, c, sizeof(double) * N * N);

        // Write ground truth time to CSV
        fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d\n", N, "ijk", "Serial", 1, time_ground_truth, 1.0, 0);
        fflush(csv_file); // Force write to CSV

        // Loop over all permutations
        for (int idx = 0; idx < 6; idx++)
        {
            // Reset c
            memset(c, 0, sizeof(c));

            // Serial version
            printf("\nStarting serial %s multiplication...\n", func_names[idx]);
            start = omp_get_wtime();
            serial_funcs[idx]();
            end = omp_get_wtime();
            time_serial = end - start;
            printf("Serial %s time: %lf seconds\n", func_names[idx], time_serial);

            // Validate against ground truth
            validate_results(func_names[idx], c, c_gt);

            // Write serial results to CSV
            fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d\n", N, func_names[idx], "Serial", 1, time_serial, time_ground_truth / time_serial, 0);
            fflush(csv_file); // Force write to CSV

            // Test with different thread counts
            for (int t_idx = 0; t_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); t_idx++)
            {
                int nthreads = thread_counts[t_idx];
                printf("\nStarting parallel %s multiplication with %d threads...\n", func_names[idx], nthreads);

                // Reset c2
                memset(c2, 0, sizeof(c2));

                start = omp_get_wtime();
                parallel_funcs[idx](nthreads, chunk);
                end = omp_get_wtime();
                time_parallel = end - start;
                printf("Parallel %s time with %d threads: %lf seconds\n", func_names[idx], nthreads, time_parallel);
                printf("Speedup: %2.2lf\n", time_serial / time_parallel);

                // Validate against ground truth
                validate_results(func_names[idx], c2, c_gt);

                // Write parallel results to CSV
                fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d\n", N, func_names[idx], "Parallel", nthreads, time_parallel, time_serial / time_parallel, 0);
                fflush(csv_file); // Force write to CSV
            }
        }

        // Blocked algorithms
        int block_sizes[] = {16, 32, 64, 128, 256};
        for (int bs_idx = 0; bs_idx < sizeof(block_sizes) / sizeof(block_sizes[0]); bs_idx++)
        {
            BS = block_sizes[bs_idx];
            printf("\nTesting block size: %d\n", BS);

            // Reset c
            memset(c, 0, sizeof(c));

            // Serial blocked multiplication
            printf("Starting serial blocked multiplication...\n");
            start = omp_get_wtime();
            serial_blocked_multiply(BS);
            end = omp_get_wtime();
            time_serial = end - start;
            printf("Serial blocked time: %lf seconds\n", time_serial);

            // Validate against ground truth
            validate_results("Blocked Serial", c, c_gt);

            // Write serial blocked results to CSV
            fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d\n", N, "Blocked", "Serial", 1, time_serial, time_ground_truth / time_serial, BS);
            fflush(csv_file); // Force write to CSV

            // Test with different thread counts
            for (int t_idx = 0; t_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); t_idx++)
            {
                int nthreads = thread_counts[t_idx];
                printf("\nStarting parallel blocked multiplication with %d threads...\n", nthreads);

                // Reset c2
                memset(c2, 0, sizeof(c2));

                start = omp_get_wtime();
                parallel_blocked_multiply(BS, nthreads);
                end = omp_get_wtime();
                time_parallel = end - start;
                printf("Parallel blocked time with %d threads: %lf seconds\n", nthreads, time_parallel);
                printf("Speedup: %2.2lf\n", time_serial / time_parallel);

                // Validate against ground truth
                validate_results("Blocked Parallel", c2, c_gt);

                // Write parallel blocked results to CSV
                fprintf(csv_file, "%d,%s,%s,%d,%lf,%lf,%d\n", N, "Blocked", "Parallel", nthreads, time_parallel, time_serial / time_parallel, BS);
                fflush(csv_file); // Force write to CSV
            }
        }
    }

    fclose(csv_file);
    printf("\nPerformance data written to matrix_mult_results.csv\n");
    return 0;
}

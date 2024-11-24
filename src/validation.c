/* src/validation.c */
#include "validation.h"
#include <stdio.h>
#include <math.h>

void validate_results(const char *test_name, double mat[MAX_SIZE][MAX_SIZE], double mat_gt[MAX_SIZE][MAX_SIZE]) {
    int mismatch = 0;
    for (int i = 0; i < N && !mismatch; i++)
        for (int j = 0; j < N; j++)
            if (fabs(mat[i][j] - mat_gt[i][j]) > EPSILON) {
                mismatch = 1;
                break;
            }
    if (mismatch)
        printf("Mismatch in %s results!\n", test_name);
    else
        printf("%s results are correct.\n", test_name);
}

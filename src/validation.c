/* src/validation.c */
#include "validation.h"
#include <stdio.h>
#include <math.h>

int validate_results(const char *test_name, double **mat, double **mat_gt, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (fabs(mat[i][j] - mat_gt[i][j]) > EPSILON)
            {
                printf("Mismatch in %s results at position [%d][%d]: %lf != %lf\n", test_name, i, j, mat[i][j], mat_gt[i][j]);
                return 0;
            }
    printf("%s results are correct.\n", test_name);
    return 1;
}

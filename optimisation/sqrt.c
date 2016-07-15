#include "sqrt.h"
#include <math.h>
#include <omp.h>

void c_sqrt(const double * matrix,
            const unsigned long long int n_rows,
            const unsigned long long int n_cols,
            double * output) {

    unsigned long long int i, j;

#pragma omp parallel for private (i, j)
    for(i=0; i<n_rows; i++){
        int n_threads = omp_get_num_threads();
        printf("%d\n",n_threads);
        for(j=0; j<n_cols; j++){
            output[i*n_cols + j] = sqrt(matrix[i*n_cols + j]);
        }
    }
}
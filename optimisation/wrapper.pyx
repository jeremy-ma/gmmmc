import numpy as np
cimport numpy as np

cdef extern from "sqrt.h":
    void c_sqrt(const double * matrix,
                const unsigned long long int n_rows,
                const unsigned long long int n_cols,
                double * output)

def c_wrapped_sqrt(np.ndarray[np.double_t, ndim=2, mode="c"] matrix):
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] to_return
    to_return = np.empty_like(matrix)

    c_sqrt(&matrix[0,0], matrix.shape[0], matrix.shape[1], &to_return[0,0])
    return to_return
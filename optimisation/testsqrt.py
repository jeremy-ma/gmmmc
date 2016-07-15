import numpy as np
import sqrt

matrix_size = 600

np.random.seed(3435)
m = np.random.uniform(size=(matrix_size, matrix_size))

def numpy_call():
    return np.sqrt(m)

def c_call():
    return sqrt.c_wrapped_sqrt(m)


numpy_call()

c_call()
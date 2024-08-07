def estimate_moment(long[:] moment, double[::1,:] X):
    cdef double result = 0.0
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t order = moment.shape[0]
    cdef double curr_est
    for i in range(n):
        curr_est = 1.0
        for fac in range(order):
            curr_est = curr_est * X[i, moment[fac]]
        result += curr_est
    return result/n

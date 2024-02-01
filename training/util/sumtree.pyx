import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from cython cimport boundscheck, wraparound

cdef class SumTree:
    """
    Sumtree that stores the weights of the transitions.
    Support fast sampling and updating via tree structure.
    Support batch operations optimized in C.
    """
    cdef long capacity
    cdef long write
    cdef double[:] data, tree

    def __init__(self, long capacity):
        self.capacity = capacity
        self.write = 0
        self.data = np.zeros(capacity, dtype=np.float64)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    @boundscheck(False)
    cdef void __update(self, long dataIdx, double weight) noexcept nogil:
        cdef long idx = dataIdx + self.capacity - 1
        cdef double change = weight - self.tree[idx]
        self.tree[idx] = weight
        self._vectorized_propagate(idx, change)
        self.data[dataIdx] = weight

    @boundscheck(False)
    cdef void _vectorized_propagate(self, const long idx, double change) noexcept nogil:
        cdef long id = (idx - 1) // 2
        while id != 0:
            self.tree[id] += change
            id = (id - 1) // 2
        self.tree[0] += change

    @boundscheck(False)
    cdef long _retrieve(self, double s, long idx) noexcept nogil:
        cdef long left = 2 * idx + 1
        cdef long right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self.tree[left], right)

    def add(self, double weight):
        self.update(self.write, weight)
        self.write = (self.write + 1) % self.capacity

    def update(self, long dataIdx, double weight):
        self.__update(dataIdx, weight)

    def get_index(self, double s):
        cdef long idx
        with nogil:
            idx = self._retrieve(s, 0)
        cdef long dataIdx = idx - self.capacity + 1
        return dataIdx

    def get_index_data(self, double s):
        cdef long idx
        with nogil:
            idx = self._retrieve(s, 0)
        cdef long dataIdx = idx - self.capacity + 1
        return dataIdx, self.data[dataIdx]

    def total(self):
        return self.tree[0]

    def __getitem__(self, idx: long):
        cdef long i = idx
        return self.data[i]

    @boundscheck(False)
    @wraparound(False)
    def batch_update(self, long[:] dataIdxs, double[:] weights):
        cdef long n = dataIdxs.shape[0]
        cdef long i
        with nogil:
            for i in prange(n):
                self.__update(dataIdxs[i], weights[i])

    @boundscheck(False)
    @wraparound(False)
    def batch_get_index(self, double[:] values):
        cdef long n = values.shape[0]
        cdef long[:] indices = np.empty(n, dtype=np.int64)
        cdef long i
        for i in range(n):
            indices[i] = self.get_index(values[i])
        return indices

    @boundscheck(False)
    @wraparound(False)
    def batch_get_index_data(self, double[:] values):
        cdef long n = values.shape[0]
        cdef long[:] indices = np.empty(n, dtype=np.int64)
        cdef double[:] data = np.empty(n, dtype=np.float64)
        cdef long i
        for i in range(n):
            indices[i] = self.get_index(values[i])
            data[i] = self.data[indices[i]]
        return indices, data

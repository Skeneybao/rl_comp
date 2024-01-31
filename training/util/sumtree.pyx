import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from cython cimport boundscheck, wraparound

cdef class SumTree:
    cdef int capacity
    cdef int write
    cdef float[:] data, tree

    def __init__(self, int capacity):
        self.capacity = capacity
        self.write = 0
        self.data = np.zeros(capacity, dtype=np.float32)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)

    @boundscheck(False)
    cdef void __update(self, int dataIdx, float weight) noexcept nogil:
        cdef int idx = dataIdx + self.capacity - 1
        cdef float change = weight - self.tree[idx]
        self.tree[idx] = weight
        self._vectorized_propagate(idx, change)
        self.data[dataIdx] = weight

    @boundscheck(False)
    cdef void _vectorized_propagate(self, const int idx, float change) noexcept nogil:
        cdef int id = (idx - 1) // 2
        while id != 0:
            self.tree[id] += change
            id = (id - 1) // 2
        self.tree[0] += change

    @boundscheck(False)
    cdef int _retrieve(self, float s, int idx) noexcept nogil:
        cdef int left = 2 * idx + 1
        cdef int right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(s, left)
        else:
            return self._retrieve(s - self.tree[left], right)

    def add(self, float weight):
        self.update(self.write, weight)
        self.write = (self.write + 1) % self.capacity

    def update(self, int dataIdx, float weight):
        self.__update(dataIdx, weight)

    def get_index(self, float s):
        cdef int idx
        with nogil:
            idx = self._retrieve(s, 0)
        cdef int dataIdx = idx - self.capacity + 1
        return dataIdx

    def get_index_data(self, float s):
        cdef int idx
        with nogil:
            idx = self._retrieve(s, 0)
        cdef int dataIdx = idx - self.capacity + 1
        return dataIdx, self.data[dataIdx]

    def total(self):
        return self.tree[0]

    def __getitem__(self, idx: int):
        cdef int i = idx
        return self.data[i]

    @boundscheck(False)
    @wraparound(False)
    def batch_update(self, int[:] dataIdxs, double[:] weights):
        cdef int n = dataIdxs.shape[0]
        cdef int i
        with nogil:
            for i in prange(n):
                self.__update(dataIdxs[i], weights[i])

    @boundscheck(False)
    @wraparound(False)
    def batch_get_index(self, float[:] values):
        cdef int n = values.shape[0]
        cdef int[:] indices = np.empty(n, dtype=np.int32)
        cdef int i
        for i in range(n):
            indices[i] = self.get_index(values[i])
        return indices

    @boundscheck(False)
    @wraparound(False)
    def batch_get_index_data(self, double[:] values):
        cdef int n = values.shape[0]
        cdef int[:] indices = np.empty(n, dtype=np.int32)
        cdef float[:] data = np.empty(n, dtype=np.float32)
        cdef int i
        for i in range(n):
            indices[i] = self.get_index(values[i])
            data[i] = self.data[indices[i]]
        return indices, data

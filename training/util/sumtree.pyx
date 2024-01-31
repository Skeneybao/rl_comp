import numpy as np
cimport numpy as cnp

cdef class SumTree:
    cdef int capacity
    cdef int write
    cdef cnp.ndarray data, tree

    def __init__(self, int capacity):
        self.capacity = capacity
        self.write = 0
        self.data = np.zeros(capacity, dtype=np.float32)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)

    cdef void _propagate(self, int idx, float change):
        cdef int parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    cdef int _retrieve(self, float s, int idx):
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
        cdef int idx = dataIdx + self.capacity - 1
        cdef float change = weight - self.tree[idx]
        self.tree[idx] = weight
        self._propagate(idx, change)
        self.data[dataIdx] = weight

    def get_index(self, float s):
        cdef int idx = self._retrieve(s, 0)
        cdef int dataIdx = idx - self.capacity + 1
        return dataIdx

    def get_index_data(self, float s):
        cdef int idx = self._retrieve(s, 0)
        cdef int dataIdx = idx - self.capacity + 1
        return dataIdx, self.data[dataIdx]

    def total(self):
        return self.tree[0]

    def __getitem__(self, idx):
        return self.data[idx]

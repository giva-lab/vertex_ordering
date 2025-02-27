import numpy as np
from scipy.sparse import csr_matrix

def get_topological_measure_optimized(A, order):
    # Get nonzero indices and values
    rows, cols = A.nonzero()
    order = np.array(order)
    # Get order differences for all nonzero connections
    diffs = np.abs(order[cols] - order[rows])
    # Sum connections per row
    connections = np.diff(A.indptr)
    # Reshape and get maximum differences per row
    max_diffs = np.zeros(A.shape[0])
    np.maximum.at(max_diffs, rows, diffs)
    # Compute the result as per formula
    res = max_diffs / connections
    return res

def get_topological_measure(A, order):
    res = []
    for i in range(0, A.shape[0]):
        row = (A[[i]].toarray() > 0)*1
        value = abs((row[0]*order)[row[0] > 0] - order[i]).max() / sum(row[0])
        res.append(value)
    return res
import numpy as np
import pandas
import os
import copy
import warnings
# import matlab.engine
from cssp import column_selection_sp

engine_started = False
eng = None

def column_selection_kppSample(mat,k=10,axis=1,randomly=False):
    """Iteratively select independent columns/rows with large norm. Once column
    c has been selected, the whole matrix will be projected along the direction
    of column c, such that no other dependent columns will be selected lately.

    Keyword arguments:
    M -- the matrix whose columns or rows to be selected from.
    k -- the number of columns/rows to be selected (default 10).
    axis -- 1 for columns, 0 for rows (default 1).
    randomly -- if false, always select the column/rows with the largest norm from
        thr remaining matrix; if true, select with probability proportional to
        their squared norms.
    """
    M = copy.deepcopy(mat)
    if axis == 0:
        M = M.T
    n, d = M.shape
    sampled_idx = []
    max_norm2 = None  # for numeric error
    for i in range(k):
        squared_norms = np.sum(np.multiply(M, M), axis=0)
        if randomly:
            new_idx = np.random.choice(d, size=1, p=squared_norms/squared_norms.sum())
        else:
            new_idx = np.argmax(squared_norms)
        if max_norm2 is None:
            max_norm2 = squared_norms.max()  # for numericly check if any norm is almost 0.

        if squared_norms[0, new_idx] == 0 or squared_norms[0, new_idx] / max_norm2 < 1e-15:
            warnings.warn("Rank < k.", RuntimeWarning)
            sampled_idx.sort()
            if axis == 1:
                Ms = mat[:, sampled_idx]
            else:
                Ms = mat[sampled_idx, :]
            return Ms, sampled_idx
        sampled_idx.append(new_idx)
        # eliminate selected column from the whole matrix.
        u = M[:,[new_idx]]
        u /= np.linalg.norm(u)
        M -= u.dot(u.T.dot(M))

    sampled_idx.sort()
    if axis == 1:
        Ms = mat[:, sampled_idx]
    else:
        Ms = mat[sampled_idx, :]
    return Ms, sampled_idx


def column_selection_leverage(mat, d, k):
    # Leverage score sampling
    MMTI = np.linalg.pinv(np.dot(mat, mat.T))
    leverage_score = np.sum(mat * np.dot(MMTI, mat), axis=0)
    print('The leverage scores:', leverage_score)
    p = leverage_score / np.sum(leverage_score)
    leverage_idx = np.random.choice(d, size=k, replace=False, p=p)
    leverage_idx.sort()
    print('Leverage - Sampled column indexes:', leverage_idx)
    Ms = mat[:, leverage_idx]
    return Ms, leverage_idx


def column_selection_norm(mat, d, k):
    norms = np.linalg.norm(mat, axis=0)
    squared_norms = norms * norms
    print('The squared norms:', squared_norms)
    p = squared_norms / np.sum(squared_norms)
    norm_idx = np.random.choice(d, size=k, replace=False, p=p)
    norm_idx.sort()
    print('Norm - Sampled column indexes:', norm_idx)
    Ms = mat[:, norm_idx]
    return Ms, norm_idx


def column_selection_uniform(mat, d, k):
    random_idx = np.random.choice(d, size=k, replace=False)
    random_idx.sort()
    print('Uniform - Sampled column indexes:', random_idx)
    Ms = mat[:, random_idx]
    return Ms, random_idx


# def column_selection_matlab(mat, d, k):
#     global engine_started, eng
#     if not engine_started:
#         eng = matlab.engine.start_matlab()
#         eng.addpath("./cssp/", nargout=0)
#         engine_started = True
#     np.savetxt("./cssp/CSSP_Input_Table.txt", mat, fmt="%.6f", delimiter=",")
#     eng.iter_fs("./cssp/CSSP_Input_Table.txt", "./cssp/CSSP_Output_Table.txt", k, k, 1e3, 1e-10)
#     results = pandas.read_table("./cssp/CSSP_Output_Table.txt", sep=",")
#     os.system("rm ./cssp/CSSP_*.txt")
#     matlab_idx = np.asarray(results.values[0, 2:k+2], dtype=int).tolist()
#     matlab_idx = [i-1 for i in matlab_idx]
#     matlab_idx.sort()
#     Ms = mat[:, matlab_idx]
#     return Ms, matlab_idx


def column_selection_hack(mat, d, k):
    path = "./hack_indices/selected_index_"
    with open(path + str(k) + ".txt", "r") as f:
        hack_idx = np.asarray([int(i) for i in list(f.readlines())])
        Ms = mat[:, hack_idx]
        return Ms, hack_idx


def column_selection(mat, d, k, mode='default', seed=31):
    if mode == 'default' or mode == 'uniform':
        return column_selection_uniform(mat, d, k)
    # elif mode == 'norm':
    #     return column_selection_norm(mat, d, k)
    # elif mode == 'leverage':
    #     return column_selection_leverage(mat, d, k)
    # elif mode == 'matlab':
    #     return column_selection_matlab(mat, d, k)
    elif mode == 'CSS-LSS':
        return column_selection_kppSample(mat, k, axis=1)
    elif mode == 'hack':
        return column_selection_hack(mat, d, k)
    elif mode == "CSS-IFS":
        return column_selection_sp(mat, d, k, seed=seed)
    else:
        print("No valid mode is chosen. Will use uniform selection.")
        return column_selection_uniform(mat, d, k)

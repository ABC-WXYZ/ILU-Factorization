# ILUT(tau, p) : A DUAL THRESHOLD INCOMPLETE FACTORIZATION, based on the paper by Y. Saad
# DATE : 8 NOVEMBER 2025
 
import ILU_Lib as libilu
from scipy import linalg
import numpy as np
import time

def sort1(t):  
    return -abs(t[0])

def sort2(t):
    return t[1]

def drop_with_threshold(row, num_to_keep, tau, st, en):
    # in the sub-row starting at index 'st' and going to index 'en - 1',
    # only keep at most num_to-keep elements which are MORE THAN tau * (norm2 of the sub-row) 

    norm = libilu.norm2(row, st, en)
    ls = [] # holds all elems more than tau * (norm2 of the sub-row)
    # print(f'The dropping threshold is {tau * norm}')
    for i in range(st, en):
        if abs(row[i]) > tau * norm:
            ls.append((row[i], i))

    if len(ls) > num_to_keep:
        ls.sort(key=sort1)
        ls = ls[:min(num_to_keep, len(ls))]

    ls.sort(key=sort2)

    indices_to_keep = set()
    for p in ls: indices_to_keep.add(p[1])

    for i in range(st, en):
        if i not in indices_to_keep:
            row[i] = 0

def ILUT(A_, tau, p):
    # Arguments:
    # A_ is the matrix to factorise in list-of-lists form
    # tau is the dropping threshold (float)
    # p is the max fill-in (integer, 1 <= p <= n)

    A = [row[:] for row in A_] # copy of the input
    n = len(A)
    original = [row[:] for row in A_] # copy of the input
    st = time.time()

    # initialize L and U matrices
    L = libilu.IdentityMatrix(n)
    U = libilu.IdentityMatrix(n, diag=0)
    U[0] = A[0].copy() # trivially, the first row of U is the same as A

    # begin the main loop
    for i in range(0, n):
        w = A[i].copy() # initialize working row
        norm_wl = libilu.norm2(w, 0, i) # 2-norm of the L part of w

        # now, create one-by-one the zeros in w
        for k in range(0, i):
            if w[k] != 0:
                if abs(w[k]) < tau * norm_wl:
                    w[k] = 0
                else:
                    w[k] = w[k] / U[k][k] # create piv

                    # now do the subtractions to create the U part of w
                    for j in range(k + 1, n):
                        w[j] = w[j] - (w[k] * U[k][j])

        # having created the L and U parts of w, do the dropping
        drop_with_threshold(w, p - 1, tau, 0, i) # L part
        drop_with_threshold(w, p - 1, tau, i + 1, n) # U part

        # write the entries in L and U
        for j in range(0, i): L[i][j] = w[j]
        for j in range(i, n): U[i][j] = w[j]

    en = time.time()
    exec_time = en - st
    print(f'ILU factorization with tau = {tau} and max_fill-in = {p}:')
    print("The execution time is", exec_time, "seconds")

    print("The L matrix is: ")
    print(L)
    print("The U matrix is: ")
    print(U)   

    LU = np.dot(np.array(L), np.array(U))
    E = np.array(original) - LU
    err = linalg.norm(E, 1)
    print("The error norm is", err)

    libilu.matrix_plotter(libilu.sum(libilu.to_binary(L), libilu.to_binary(A)))   


M = [
        [4.0, -1.0, 0.0,  0.0,  0.0,  0.0],
        [-1.0, 4.0, -1.0, 0.0,  0.0,  0.0],
        [0.0, -1.0, 4.0, -1.0,  0.0,   0.0],
        [27.0,  0.0, -1.0, 4.0, -1.0,  0.0],
        [0.0,  13.0,  0.0, -1.0, 4.0, -1.0],
        [0.0,  0.0,  0.0,  0.0,-1.0, 4.0]
]

ILUT(libilu.bcsstk01, 1e-7, 48)
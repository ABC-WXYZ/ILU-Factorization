import ILU_Lib as libilu
import time
import numpy as np
import scipy as sp
# This file contains the code for ILU0 factorization. 

matrix = [
    [4, 1, 0],
    [1, 4, 1],
    [0, 1, 4]
]

def ILU0(A, n):
    # A is a n x n matrix
    original = [row[:] for row in A]
    st = time.time()
    L = [0] * n
    for i in range(n):
        L[i] = [0] * n
    for i in range(n): L[i][i] = 1

    for i in range(1, n):
        for k in range(0, i):
            if (A[i][k] != 0):
                piv = A[i][k] / A[k][k]
                L[i][k] = piv
                for j in range(k, n):
                    if (A[i][j] != 0) :
                        A[i][j] = A[i][j] - piv * A[k][j]
                        
    en = time.time()
    exec_time = en - st
    print("ILU(0) Factorization: ")
    print("The execution time is", exec_time, "seconds")
    LU = np.dot(np.array(L), np.array(A))
    E = np.array(original) - LU
    err = sp.linalg.norm(E, 1)
    print("The error norm is", err)

    # print("The L matrix is: ")
    # libilu.display(L)
    # print("The U matrix is: ")
    # libilu.display(A)   
    libilu.matrix_plotter(libilu.sum(libilu.to_binary(L), libilu.to_binary(A))) 

A = libilu.bcsstk01
ILU0(A, 48)
print()



# [
#     [4, -1,  0, -1,  0,  0,  0,  0,  0],
#     [-1, 4, -1,  0, -1,  0,  0,  0,  0],
#     [0, -1,  4,  0,  0, -1,  0,  0,  0],
#     [-1, 0,  0,  4, -1,  0, -1,  0,  0],
#     [0, -1,  0, -1,  4, -1,  0, -1,  0],
#     [0,  0, -1,  0, -1,  4,  0,  0, -1],
#     [0,  0,  0, -1,  0,  0,  4, -1,  0],
#     [0,  0,  0,  0, -1,  0, -1,  4, -1],
#     [0,  0,  0,  0,  0, -1,  0, -1,  4]
# ]
# [
#     [6, -2, 2, 4],
#     [12, -8, 6, 10],
#     [3, -13, 9, 3],
#     [-6, 4, 1, 18]
# ]
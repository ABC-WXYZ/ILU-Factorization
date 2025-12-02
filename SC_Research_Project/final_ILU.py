import ILU_Lib as libilu
from scipy import linalg
import numpy as np
import time

# This file contains code for Simple LU factorization and leevel-based ILU factorization.

def Simple_LU(A):
    # A is assumed to be a square matrix
    n = len(A)
    for i in range(1, n):
        for k in range(0, i):
            A[i][k] = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]

    L = libilu.IdentityMatrix(n)
    for i in range(n):
        for j in range(i):
            L[i][j] = A[i][j]
            A[i][j] = 0

    # print("Simple LU Factorization: ")
    # print("The L matrix is: ")
    # libilu.display(L)
    # print("The U matrix is: ")
    # libilu.display(A)   
    libilu.matrix_plotter(libilu.sum(libilu.to_binary(L), libilu.to_binary(A))) 

def ILUP(A, p):
    # ILU with p fill-in

    # st time
    n = len(A)
    original = [row[:] for row in A]
    st = time.time()

    level = libilu.IdentityMatrix(n, diag=0)
    for i in range(n):
        for j in range(n):
            if((i == j) or (A[i][j] != 0)): level[i][j] = 0
            else: level[i][j] = float('inf')

    for i in range(1, n):
        for k in range(0, i):
            A[i][k] = A[i][k] / A[k][k]
            for j in range(k + 1, n):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
                level[i][j] = min(level[i][j], level[i][k] + level[k][j] + 1)
                if level[i][j] > p: A[i][j] = 0

    L = libilu.IdentityMatrix(n)
    for i in range(n):
        for j in range(i):
            L[i][j] = A[i][j]
            A[i][j] = 0

    en = time.time()
    exec_time = en - st
    print(f'ILU factorization with max level = {p}:')
    print("The execution time is", exec_time, "seconds")

    # print("The L matrix is: ")
    # libilu.display(L)
    # print("The U matrix is: ")
    # libilu.display(A)   

    LU = np.dot(np.array(L), np.array(A))
    E = np.array(original) - LU
    err = linalg.norm(E, 1)
    print("The error norm is", err)

    libilu.matrix_plotter(libilu.sum(libilu.to_binary(L), libilu.to_binary(A)))



# A = [  [ 3,  0, -1, -1,  0, -1 ],
#        [ 0,  2,  0, -1,  0,  0 ],
#        [-1,  0,  3,  0, -1,  0 ],
#        [-1, -1,  0,  2,  0, -1 ],
#        [ 0,  0, -1,  0,  3, -1 ],
#        [-1,  0,  0, -1, -1,  4 ]
#     ]

# # Simple_LU(A)
# # print()
# print("What happened to A? ")
# libilu.display(A)
# print()
A = libilu.bcsstk01
Simple_LU(A)


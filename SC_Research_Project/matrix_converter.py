import ILU_Lib as libilu

M = []
nnz = 0
with open("sherman2.txt") as f:
    details_read = False
    for line in f:
        if not details_read:
            n = (line.strip()).split()[0]
            print("The size of matrix is", n)
            M = libilu.IdentityMatrix(int(n), diag=0)
            # The file is assumed to contain a square matrix
            details_read = True
        else:
            element = (line.strip()).split()
            i = int(element[0])
            j = int(element[1])
            val = float(element[2])
            M[i - 1][j - 1] = val 
            nnz += 1
            # if i != j:
            #     M[j - 1][i - 1] = val # For symmetric matrices
            #     nnz += 1

print(f'There are {nnz} non-zero entries.')
# libilu.display(M)

with open("sherman2_py.txt", "w") as f:
    print(M, file=f)
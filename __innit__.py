import numpy as np

def add(a,b):
    return a+b

def lerp(a,b,t):
    return a*(1-t)+b*(t)
    

def linear_bezier(a,b,t):
    return np.array(lerp(a,b,t))

def quadratic_bezier(a,b,c,t):
    return np.array((1-t)**2*a+2*(1-t)*t*b+t**2*c)

def cubic_bezier(a,b,c,d,t):
    return np.array((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t**2*c+t**3*d)


def matrix_inversion(A):
    n = A.shape[0]
    A = np.array(A, dtype=float)
    I = np.eye(n)
    augmented = np.hstack((A, I))

    for i in range(n):
        max_row = np.argmax(abs(augmented[i:, i])) + i
        if i != max_row:
            augmented[[i, max_row]] = augmented[[max_row, i]]
        
        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot
        
        for j in range(n):
            if i != j:
                row_factor = augmented[j, i]
                augmented[j] -= row_factor * augmented[i]

    inverse = augmented[:, n:]
    return inverse

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        L[i, i] = 1
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

def qr_decomposition(A):
    n, m = A.shape
    Q = np.zeros((n, n))
    R = np.zeros((n, m))

    for i in range(m):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    
    return Q, R



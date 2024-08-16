import numpy as np
import cmath
import scipy.integrate as integrate
import scipy.signal as signal
import pywt
from scipy.signal import hilbert
import numpy.polynomial.legendre as leg

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

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
    
    return L

def power_iteration(A, num_simulations=1000):
    n, _ = A.shape
    b_k = np.random.rand(n)

    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    eigenvector = b_k
    return eigenvalue, eigenvector

def eigen_decomposition(A, tol=1e-10):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        eigenvalue, eigenvector = power_iteration(A)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        A = A - eigenvalue * np.outer(eigenvector, eigenvector)
        if np.linalg.norm(A) < tol:
            break
    
    return np.array(eigenvalues), np.array(eigenvectors).T

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]

def ifft(X):
    N = len(X)
    if N <= 1:
        return X
    even = ifft(X[0::2])
    odd = ifft(X[1::2])
    T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    result = [even[k] + T[k] for k in range(N // 2)] + \
             [even[k] - T[k] for k in range(N // 2)]
    return [x / 2 for x in result]

def to_polar(z):
    r = abs(z)
    theta = cmath.phase(z)
    return r, theta

def to_rectangular(r, theta):
    return r * np.exp(1j * theta)

def complex_roots(coefficients, tol=1e-10, max_iter=1000):
    n = len(coefficients) - 1
    roots = []
    A = np.array(coefficients, dtype=complex)
    
    for _ in range(n):
        x0 = np.random.rand() + 1j * np.random.rand()
        for _ in range(max_iter):
            P = np.polyval(A, x0)
            P_prime = np.polyval(np.polyder(A), x0)
            if abs(P_prime) < tol:
                break
            x1 = x0 - P / P_prime
            if abs(x1 - x0) < tol:
                break
            x0 = x1
        
        roots.append(x0)
        A = np.polydiv(A, np.array([1, -x0]))[0]
    
    return np.array(roots)

def contour_integral(f, C, num_points=1000):
    t = np.linspace(0, 1, num_points)
    z = np.array([C(s) for s in t])
    dz_dt = np.gradient(z, t)

    integrand = np.array([f(z[i]) * dz_dt[i] for i in range(num_points)])
    result = integrate.simps(integrand, t)
    
    return result

def complex_derivative(f, z, h=1e-10):
    return (f(z + h) - f(z - h)) / (2 * h)

def matrix_convolution(A, B):
    return signal.convolve2d(A, B, mode='full')

def dwt(data, wavelet='haar', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def idwt(coeffs, wavelet='haar'):
    reconstructed_data = pywt.waverec(coeffs, wavelet)
    return reconstructed_data

def pca(data, num_components):
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean
    covariance_matrix = np.cov(centered_data.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    principal_components = eigenvectors[:, :num_components]
    reduced_data = np.dot(centered_data, principal_components)
    
    return reduced_data, principal_components

def complex_power_iteration(A, num_simulations=1000, tol=1e-10):
    n, _ = A.shape
    b_k = np.random.rand(n) + 1j * np.random.rand(n)

    for _ in range(num_simulations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        if np.linalg.norm(b_k1 - b_k * b_k1_norm) < tol:
            break
    
    eigenvalue = np.dot(b_k.T.conjugate(), np.dot(A, b_k)) / np.dot(b_k.T.conjugate(), b_k)
    eigenvector = b_k
    return eigenvalue, eigenvector


def hilbert_transform(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    instantaneous_frequency = np.diff(np.unwrap(instantaneous_phase))
    
    return amplitude_envelope, instantaneous_phase, instantaneous_frequency

def gaussian_quadrature(f, a, b, n):
    x, w = leg.leggauss(n)
    t = 0.5 * (x + 1) * (b - a) + a
    integral = sum(w * f(t)) * 0.5 * (b - a)
    return integral

def complex_conjugate_transpose(A):
    return np.conjugate(A.T)


def nmf(V, num_components, num_iterations=1000, tol=1e-4):
    n, m = V.shape
    W = np.abs(np.random.randn(n, num_components))
    H = np.abs(np.random.randn(num_components, m))

    for _ in range(num_iterations):
        H = H * (np.dot(W.T, V) / (np.dot(W.T, np.dot(W, H)) + tol))
        W = W * (np.dot(V, H.T) / (np.dot(W, np.dot(H, H.T)) + tol))
    
    return W, H

def kalman_filter(F, H, Q, R, x0, P0, measurements):
    x = x0
    P = P0
    x_estimates = []

    for z in measurements:
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        y = z - np.dot(H, x)
        S = np.dot(H, np.dot(P, H.T)) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        
        x = x + np.dot(K, y)
        P = np.dot(np.eye(len(x)) - np.dot(K, H), P)
        
        x_estimates.append(x)
    
    return np.array(x_estimates)

def svd_image_compression(image, num_components):
    U, S, V = np.linalg.svd(image, full_matrices=False)
    
    S_reduced = np.zeros((num_components, num_components))
    np.fill_diagonal(S_reduced, S[:num_components])
    
    compressed_image = np.dot(U[:, :num_components], np.dot(S_reduced, V[:num_components, :]))
    return compressed_image

def complex_modulus_argument(z):
    modulus = abs(z)
    argument = np.angle(z)
    return modulus, argument

def linear_least_squares(A, b):
    A_conj_trans = np.conjugate(A.T)
    x = np.linalg.solve(np.dot(A_conj_trans, A), np.dot(A_conj_trans, b))
    return x

def schur_decomposition(A):
    T, Z = np.linalg.schur(A)
    return T, Z

def complex_gradient_descent(f, grad_f, z0, learning_rate=0.01, num_iterations=1000):
    z = z0
    for _ in range(num_iterations):
        grad = grad_f(z)
        z = z - learning_rate * grad
    return z

def runge_kutta(f, y0, t0, t1, h):
    t = np.arange(t0, t1 + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(1, len(t)):
        k1 = h * f(t[i-1], y[i-1])
        k2 = h * f(t[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * f(t[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * f(t[i], y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y

def multivariate_complex_gaussian(mean, cov, size=1):
    real_part = np.random.multivariate_normal(np.real(mean), np.real(cov), size)
    imag_part = np.random.multivariate_normal(np.imag(mean), np.imag(cov), size)
    return real_part + 1j * imag_part

def newton_raphson_complex(f, df, z0, tol=1e-10, max_iter=1000):
    z = z0
    for _ in range(max_iter):
        dz = f(z) / df(z)
        z = z - dz
        if abs(dz) < tol:
            break
    return z

def eigenvalue_perturbation_analysis(A, dA):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    perturbed_eigenvalues = []

    for i in range(len(eigenvalues)):
        vi = eigenvectors[:, i]
        perturbed_lambda = eigenvalues[i] + np.dot(np.dot(vi.T.conjugate(), dA), vi) / np.dot(vi.T.conjugate(), vi)
        perturbed_eigenvalues.append(perturbed_lambda)
    
    return np.array(perturbed_eigenvalues)

def solve_complex_riccati(A, B, C, D):
    X = np.zeros_like(A, dtype=complex)
    max_iter = 100
    tol = 1e-10

    for _ in range(max_iter):
        X_new = A + B @ X @ C + D @ X @ X
        if np.linalg.norm(X_new - X) < tol:
            break
        X = X_new
    
    return X
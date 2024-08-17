import numpy as np
import cmath
import scipy.integrate as integrate
import scipy.signal as signal
import pywt
import numpy.polynomial.legendre as leg
from sklearn.svm import SVC
from scipy.fft import fft, ifft
from scipy.stats import norm
from sklearn.decomposition import SparseCoder
import emd

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

def complex_singular_value_thresholding(M, tau, max_iter=100):
    X = np.zeros_like(M, dtype=complex)
    for _ in range(max_iter):
        U, S, V = np.linalg.svd(X, full_matrices=False)
        S = np.maximum(S - tau, 0)
        X_new = np.dot(U, np.dot(np.diag(S), V))
        if np.linalg.norm(X_new - X) < 1e-10:
            break
        X = X_new
    return X

def complex_k_means(X, k, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iter):
        distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    
    return centroids, labels

def complex_dct(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = sum(x[n] * np.cos(np.pi * k * (2*n + 1) / (2 * N)) + 1j * x[n] * np.sin(np.pi * k * (2*n + 1) / (2 * N)) for n in range(N))
    return X

def hermitian_matrix_completion(M, mask, max_iter=100):
    X = np.zeros_like(M, dtype=complex)
    for _ in range(max_iter):
        X = (M * mask) + (X * (1 - mask))
        X = (X + X.T.conjugate()) / 2
        if np.linalg.norm((X * mask) - M) < 1e-10:
            break
    return X



def complex_svm(X, y, C=1.0, kernel='linear'):
    real_part = np.hstack([np.real(X), np.imag(X)])
    clf = SVC(C=C, kernel=kernel)
    clf.fit(real_part, y)
    return clf

def complex_pcr(X, y, num_components):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    U, S, V = np.linalg.svd(X_centered, full_matrices=False)
    V = V.T[:, :num_components]
    
    Z = np.dot(X_centered, V)
    coeffs = np.linalg.lstsq(Z, y, rcond=None)[0]
    
    return V @ coeffs, X_mean


def complex_monte_carlo_integration(f, a, b, num_samples=10000):
    real_samples = np.random.uniform(a.real, b.real, num_samples)
    imag_samples = np.random.uniform(a.imag, b.imag, num_samples)
    
    samples = real_samples + 1j * imag_samples
    values = np.array([f(z) for z in samples])
    
    area = (b.real - a.real) * (b.imag - a.imag)
    integral = area * np.mean(values)
    
    return integral


def complex_matrix_sign(A):
    U, S, V = np.linalg.svd(A)
    S_sign = np.sign(S)
    return np.dot(U, np.dot(np.diag(S_sign), V))


def complex_lms_filter(d, x, mu=0.01, num_iterations=1000):
    N = len(x)
    w = np.zeros(N, dtype=complex)
    for n in range(num_iterations):
        y = np.dot(w.conjugate(), x)
        e = d - y
        w = w + mu * e * x.conjugate()
    return w


def complex_hankel(c, r=None):
    if r is None:
        r = np.zeros_like(c)
    elif r[0] != c[-1]:
        raise ValueError("r[0] should be the same as c[-1]")
    A = np.zeros((len(c), len(r)), dtype=complex)
    for i in range(len(c)):
        for j in range(len(r)):
            if i + j < len(c):
                A[i, j] = c[i + j]
            else:
                A[i, j] = r[i + j - len(c) + 1]
    return A

def complex_fft_filter(signal, filter_function):
    fft_signal = fft(signal)
    filtered_fft_signal = filter_function(fft_signal)
    filtered_signal = ifft(filtered_fft_signal)
    return filtered_signal

def complex_givens_rotation(a, b):
    if b == 0:
        c = 1 + 0j
        s = 0 + 0j
    else:
        if abs(b) > abs(a):
            r = a / b
            s = 1 / np.sqrt(1 + abs(r)**2)
            c = s * r
        else:
            r = b / a
            c = 1 / np.sqrt(1 + abs(r)**2)
            s = c * r
    
    return c, s

def apply_givens_rotation(A, c, s, i, j):
    for k in range(A.shape[1]):
        temp = c * A[i, k] - s * A[j, k]
        A[j, k] = s * A[i, k] + c * A[j, k]
        A[i, k] = temp
    return A

def complex_ar_model_fit(signal, p):
    N = len(signal)
    X = np.array([signal[i:N-p+i] for i in range(p)]).T
    y = signal[p:]
    
    X_pseudo_inv = np.linalg.pinv(X)
    a = np.dot(X_pseudo_inv, y)
    
    return a

def complex_ar_predict(a, signal, steps):
    p = len(a)
    predictions = []
    for _ in range(steps):
        next_value = np.dot(a, signal[-p:])
        signal = np.append(signal, next_value)
        predictions.append(next_value)
    return np.array(predictions)

def complex_wavelet_transform(signal, wavelet='cmor', level=None):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    reconstructed_signal = pywt.waverec(coeffs, wavelet=wavelet)
    return coeffs, reconstructed_signal

def complex_afd(signal, num_terms):
    N = len(signal)
    z = np.exp(1j * 2 * np.pi * np.arange(N) / N)
    coefficients = []

    for _ in range(num_terms):
        alpha = np.sum(np.conj(z) * signal) / np.sum(np.abs(z)**2)
        coefficients.append(alpha)
        signal -= alpha * z
        z *= np.exp(1j * 2 * np.pi / N)
    
    return coefficients

def complex_sparse_coding(signal, dictionary, alpha=1.0):
    real_signal = np.real(signal)
    imag_signal = np.imag(signal)
    
    coder = SparseCoder(dictionary=dictionary, transform_alpha=alpha)
    real_code = coder.transform(real_signal)
    imag_code = coder.transform(imag_signal)
    
    return real_code + 1j * imag_code

def complex_cross_correlation(x, y):
    n = len(x)
    result = np.zeros(n, dtype=complex)
    for lag in range(n):
        result[lag] = np.dot(np.conjugate(x), np.roll(y, lag))
    return result

def complex_bayesian_inference(prior_mean, prior_var, likelihood_var, data):
    posterior_mean = prior_mean
    posterior_var = prior_var
    
    for observation in data:
        likelihood_mean = observation
        posterior_var = 1 / (1/prior_var + 1/likelihood_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + likelihood_mean/likelihood_var)
        prior_mean, prior_var = posterior_mean, posterior_var
    
    return posterior_mean, posterior_var

def complex_ica(X, n_components):
    real_part = np.real(X)
    imag_part = np.imag(X)
    
    ica_real = FastICA(n_components=n_components)
    ica_imag = FastICA(n_components=n_components)
    
    sources_real = ica_real.fit_transform(real_part)
    sources_imag = ica_imag.fit_transform(imag_part)
    
    sources = sources_real + 1j * sources_imag
    return sources, ica_real, ica_imag

def complex_autocorrelation(signal):
    N = len(signal)
    autocorr = np.zeros(N, dtype=complex)
    
    for lag in range(N):
        autocorr[lag] = np.dot(np.conjugate(signal[:N-lag]), signal[lag:])
    
    return autocorr

def complex_emd(signal):
    real_part = np.real(signal)
    imag_part = np.imag(signal)
    
    imfs_real = emd.sift.sift(real_part)
    imfs_imag = emd.sift.sift(imag_part)
    
    imfs = imfs_real + 1j * imfs_imag
    return imfs

def complex_conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    if x0 is None:
        x0 = np.zeros_like(b, dtype=complex)
    if max_iter is None:
        max_iter = len(b)
    
    r = b - np.dot(A, x0)
    p = r
    x = x0
    
    for _ in range(max_iter):
        Ap = np.dot(A, p)
        alpha = np.dot(np.conjugate(r), r) / np.dot(np.conjugate(p), Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(np.conjugate(r_new), r_new) / np.dot(np.conjugate(r), r)
        p = r_new + beta * p
        r = r_new
    
    return x


def complex_qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, m), dtype=complex)
    R = np.zeros((m, n), dtype=complex)
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(np.conjugate(Q[:, i]), v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R


def complex_wavelet_packet_decomposition(signal, wavelet='cmor', max_level=None):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    nodes = wp.get_level(max_level, 'freq')
    coeffs = np.array([n.data for n in nodes])
    
    return coeffs, wp


def complex_newtons_method_polynomial(p, dp, z0, tol=1e-10, max_iter=1000):
    z = z0
    for _ in range(max_iter):
        z_next = z - p(z) / dp(z)
        if abs(z_next - z) < tol:
            break
        z = z_next
    return z

def complex_ginibre_ensemble(n):
    real_part = np.random.randn(n, n)
    imag_part = np.random.randn(n, n)
    return real_part + 1j * imag_part

def complex_de_casteljau(points, t):
    n = len(points) - 1
    new_points = np.copy(points)
    for r in range(1, n + 1):
        for i in range(n - r + 1):
            new_points[i] = (1 - t) * new_points[i] + t * new_points[i + 1]
    return new_points[0], new_points[:n]

def complex_rational_bezier(points, weights, t):
    n = len(points) - 1
    numerator = np.sum([weights[i] * points[i] * (1 - t)**(n - i) * t**i for i in range(n + 1)])
    denominator = np.sum([weights[i] * (1 - t)**(n - i) * t**i for i in range(n + 1)])
    return numerator / denominator


def complex_rational_bezier(points, weights, t):
    n = len(points) - 1
    numerator = np.sum([weights[i] * points[i] * (1 - t)**(n - i) * t**i for i in range(n + 1)])
    denominator = np.sum([weights[i] * (1 - t)**(n - i) * t**i for i in range(n + 1)])
    return numerator / denominator

def complex_bezier_length(points, num_samples=100):
    length = 0
    t_values = np.linspace(0, 1, num_samples)
    for i in range(1, num_samples):
        p1 = complex_cubic_bezier(*points, t_values[i - 1])
        p2 = complex_cubic_bezier(*points, t_values[i])
        length += np.abs(p2 - p1)
    return length


import jax.numpy as np
from jax.numpy.linalg import eigh

def make_HIPPO(N):
    P = np.sqrt(1 + 2*np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]     # Reshaping P into a column (N, 1 ) and row (1, N) vector for multiplication
    A = np.tril(A) - np.diag(np.arange(N))     # Keeping the lower triangular matrix and subtracting the diagonal values
    return -A

def make_NPLR_HIPPO(N):
    nhippo = make_HIPPO(N)

    P = np.sqrt(np.arange(N) + 0.5)             # Add in a rank 1 term. Makes it Normal.
    B = np.sqrt(2 * np.arange(N) + 1.0)

    return nhippo, P,B

def make_DPLR_HIPPO(N):
    '''Diagonalize NPLR representation'''
    A, P ,B = make_NPLR_HIPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    s_diag = np.diagonal(S)
    lambda_real = np.mean(s_diag) * np.ones_like(s_diag)

    lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B

    return lambda_real + 1j * lambda_imag, P, B, V

def init(x):
    def _init(key, shape):
        assert shape == x.shape
        return x
    return _init

def hippo_initializer(N):
    Lambda, P, B, _ = make_DPLR_HIPPO(N)
    return init(Lambda.real), init(Lambda.imag), init(P), init(B)
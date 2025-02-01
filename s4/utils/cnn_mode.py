import jax
import jax.numpy as np
from jax.scipy.signal import convolve

def causal_convolution(u, kernel_K, fft=True):

    if fft == False:
        return convolve(u, kernel_K, mode='full')[:u.shape[0]]
    else:
        # assert kernel_K.shape[0] == u.shape[0]

        padded_u = np.pad(u, (0, kernel_K.shape[0]))
        padded_kernel_k = np.pad(kernel_K, (0, u.shape[0]))

        # FFT: Transforming to frequence-domain representation

        ud = np.fft.rfft(padded_u)
        kd = np.fft.rfft(padded_kernel_k)

        # Matrix Multiplication

        out = ud * kd

        # Inverse FFT: Converts the result back into the time domain

        result = np.fft.irfft(out)

        return result[:u.shape[0]]

def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()

@jax.jit
def cauchy(v, omega, lambd):
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)

def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)

    return out.real
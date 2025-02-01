import jax
import jax.numpy as np
from jax.numpy.linalg import inv, matrix_power

def scan_SSM(A_bar, B_bar, C_bar, u, x0):
    def step(x_k_1, u_k):
        x_k = A_bar @ x_k_1 + B_bar @ u_k
        y_k = C_bar @ x_k

        return x_k, y_k

    return jax.lax.scan(step, x0, u)

def discrete_DPLR(Lambda, P,Q,B,C,step, L):

    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]

    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxsis].conj().T             # Construct A in DPLR form
    I = np.eye(N)

    A0 = (2.0 / step)*I + A                                                         # Forward Euler

    D = np.diag(1.0 / ((2.0/step) - Lambda))
    Qc = Q.conj().T.reshape(1,-1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0/(1 + (Qc @ D @ P2))) * Qc @ D)                          # Backward Euler    

    A_bar = A1 @ A0
    B_bar = 2 * A1 @ B

    C_bar = Ct @ inv(I - matrix_power(A_bar, L)).conj()
    return A_bar, B_bar, C_bar.conj()
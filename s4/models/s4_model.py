from flax import linen as nn
from jax.nn.initializers import normal
import jax.numpy as np
from s4.utils.helper import clone_layer, log_step_initializer
from s4.utils.hippo import hippo_initializer
from s4.utils.cnn_mode import kernel_DPLR, causal_convolution
from s4.utils.rnn_mode import discrete_DPLR, scan_SSM

class S4Layer(nn.Module):
    N: int                              # state dimension
    l_max: int                          # max. sequence length
    decode: bool = False                # Option to run in CNN or RNN mode
    lr =  {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1
    }

    def setup(self):
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)                # This initializes matrices A,P,B while Hippo ensures that past information is preserved in a summarized way

        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))                  # Lambda is a diagonal complex matrix that represents continous state-space system. Lambda_re and Lambda_im are real and imaginary parts of Lambda

        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im

        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))                                     # P and B are learnable parameters

        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]                                   # C is a complex vector used in computing convolution kernel

        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))        # D is also a learnable parameter. Used in modifying the output. Step is paramter that controls discretization

        if not self.decode:
            '''CNN mode for pre computing the convolution kernel and using this in parallel computation'''
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )          

        else:
            '''RNN mode for discretizing SSM and then doing sequential computation'''
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)                      # Computes the discretized versiong of SSM
            
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()

            self.ssm = ssm_var.value

            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):

        if not self.decode:
            # CNN mode
            return causal_convolution(u, self.K) + self.D * u                           # Perform convolution using kernel K and add a direct connection D*u (like residual)

        else:
            # RNN mode

            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)

            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u
            
S4Layer = clone_layer(S4Layer)

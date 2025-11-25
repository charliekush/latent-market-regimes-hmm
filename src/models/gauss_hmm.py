from typing import Dict, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


class GaussianHMM:
    def __init__(
        self,
        n_states: int,
        n_iter: int = 50,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        #number of regimes
        self.n_states: int = n_states
        #number of iterations for fitting
        self.n_iter: int = n_iter
        #tolerance for EM convergence
        self.tol: float = tol

        self.random_state: np.random.RandomState = np.random.RandomState(
            random_state)

        # initial prob distrobution
        self.pi_: Optional[Array] = None        # shape (K,)
        # transition matrix
        self.A_: Optional[Array] = None         # shape (K, K)

        self.means_: Optional[Array] = None     # shape (K,)
        self.vars_: Optional[Array] = None      # shape (K,)s

    def _log_gaussian(self, x: Array, means: Array, vars_: Array) -> Array:
        """
        x: shape (T,)
        means: shape (K,)
        vars_: shape (K,)
        returns: log B matrix, shape (T, K),
                where B[t, k] = log p(x_t | state k)
        """
        T = x.shape[0]
        K = self.n_states
        x = x[:, None]  # (T, 1)
        means = means[None, :]  # (1, K)
        vars_ = vars_[None, :]  # (1, K)

        log_norm = -0.5 * np.log(2 * np.pi * vars_)
        log_exp = -0.5 * (x - means) ** 2 / vars_

        return log_norm + log_exp  # (T, K)

    def _forward(self, log_B: Array) -> Tuple[Array, float]:
        """
        log_B: (T, K) log p(x_t | state k)
        returns: log_alpha (T, K), log_likelihood (scalar)
        """
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        eps = 1e-12
        if self.pi_ is not None:
            log_pi = np.log(self.pi_ + eps)
        else:
            raise ValueError("initial probability distribution is None")
        


        if self.A_ is not None:
            log_A = np.log(self.A_ + eps)
        else:
            raise ValueError("transition  matrix is None")

        # t=0
        log_alpha[0] = log_pi + log_B[0]

        # t>0
        for t in range(1, T):
            # over previous
            for j in range(K):
                log_alpha[t, j] = log_B[t, j] + \
                    self._logsumexp(log_alpha[t-1] + log_A[:, j])

        log_likelihood = self._logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    @staticmethod
    def _logsumexp(v: Array) -> float:
        m = np.max(v)
        return float(m + np.log(np.sum(np.exp(v - m))))

    def _backward(self, log_B: Array, log_A: Array) -> Array:
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_beta[-1] = 0.0  # log(1)

        for t in range(T - 2, -1, -1):
            for i in range(K):
                v = log_A[i, :] + log_B[t+1, :] + log_beta[t+1, :]
                log_beta[t, i] = self._logsumexp(v)

        return log_beta

    def _compute_xi(
        self,
        x: Array,
        log_B: Array,
        log_alpha: Array,
        log_beta: Array,
        log_A: Array,
        log_likelihood: float,
    ) -> Array:
        T, K = log_B.shape
        xi = np.zeros((T-1, K, K))

        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (
                        log_alpha[t, i]
                        + log_A[i, j]
                        + log_B[t+1, j]
                        + log_beta[t+1, j]
                        - log_likelihood
                    )
        return np.exp(xi)  # (T-1, K, K)

    def _m_step(self, x: Array, gamma: Array, xi: Array) -> None:
        T, K = gamma.shape
        x = x.reshape(-1)
        eps = 1e-12
        pi_raw = gamma[0] + eps          # avoid zeros
        self.pi_ = pi_raw / pi_raw.sum() # normalize
        self.pi_ = gamma[0] / gamma[0].sum()

        xi_sum = xi.sum(axis=0) + eps          # (K, K)
        gamma_sum = gamma[:-1].sum(axis=0) + eps  # (K,)
        


        A = xi_sum / gamma_sum[:, None]        # row-normalize later
        A = np.maximum(A, eps)                 # avoid exact zeros
        A = A / A.sum(axis=1, keepdims=True)   # each row sums to 1
        self.A_ = A
        

        # emmission means
        gamma_sum_all = gamma.sum(axis=0) + eps  # (K,)
        self.means_ = (gamma * x[:, None]).sum(axis=0) / gamma_sum_all

        # emission variances
        if self.means_ is not None:
            diff = x[:, None] - self.means_[None, :]
        else:
            raise ValueError("self.means_ is None")

        self.vars_ = (gamma * diff**2).sum(axis=0) / gamma_sum_all
        if self.vars_ is not None:
            self.vars_ = np.maximum(self.vars_, 1e-6)  # no zero variance
        else:
            raise ValueError("self.vars_ is None")

    def fit(self, X: Array, show_progress: bool = True) -> "GaussianHMM":
        x = X.reshape(-1)      # (T,)
        T = x.shape[0]
        K = self.n_states

        self.pi_ = np.ones(K) / K
        self.A_ = np.ones((K, K)) / K

        #  init for emissions
        self.means_ = np.linspace(x.min(), x.max(), K)
        self.vars_ = np.ones(K) * x.var()

        prev_ll = -np.inf
        progress = (
            tqdm(range(self.n_iter), desc="GaussianHMM EM", unit="iter", leave=False)
            if show_progress
            else None
        )
        iterator = progress if progress is not None else range(self.n_iter)
        try:
            for _ in iterator:
                if self.vars_ is not None and self.means_ is not None:
                    log_B = self._log_gaussian(x, self.means_, self.vars_)
                else:
                    raise ValueError("self.vars_ or self.means_is None")

                log_A = np.log(self.A_)
                log_alpha, log_ll = self._forward(log_B)
                log_beta = self._backward(log_B, log_A)

                # E-step
                log_gamma = log_alpha + log_beta - log_ll
                gamma = np.exp(log_gamma)
                xi = self._compute_xi(x, log_B, log_alpha, log_beta, log_A, log_ll)

                # M-step
                self._m_step(x, gamma, xi)

                if progress is not None:
                    progress.set_postfix(log_ll=float(log_ll))

                if np.abs(log_ll - prev_ll) < self.tol:
                    break
                prev_ll = log_ll
        finally:
            if progress is not None:
                progress.close()

        return self

    def score(self, X: Array) -> float:
        x = X.reshape(-1)
        if self.vars_ is not None and self.means_ is not None:
                log_B = self._log_gaussian(x, self.means_, self.vars_)
        else:
            raise ValueError("self.vars_ or self.means_is None")
        _, log_ll = self._forward(log_B)
        return log_ll

    def decode(self, X: Array) -> Tuple[float, IntArray]:
        x = X.reshape(-1)
        T = x.shape[0]
        K = self.n_states
        eps = 1e-12

        if self.vars_ is not None and self.means_ is not None:
                log_B = self._log_gaussian(x, self.means_, self.vars_)
        else:
            raise ValueError("self.vars_ or self.means_is None")
        

        if self.A_ is not None:
                log_A = np.log(self.A_)
        else:
            raise ValueError("self.A_ is None")
        
        if self.pi_ is not None:
                log_pi = np.log(self.pi_ + eps)
        else:
            raise ValueError("self.pi_ is None")

        log_delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        # init
        log_delta[0] = log_pi + log_B[0]

        # recursion
        for t in range(1, T):
            for j in range(K):
                vals = log_delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(vals)
                log_delta[t, j] = log_B[t, j] + vals[psi[t, j]]

        # backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        logprob = np.max(log_delta[-1])
        return logprob, states

    def summarize_regimes(self) -> Dict[str, Array]:
        if self.means_ is None or self.vars_ is None or self.A_ is None or self.pi_ is None:
            raise ValueError("Model parameters are not initialized. Call fit(...) first.")

        K = self.n_states

        means = np.asarray(self.means_).reshape(-1)        # (K,)
        vars_ = np.asarray(self.vars_).reshape(-1)         # (K,)
        stds = np.sqrt(vars_)                              # (K,)
        transmat = np.asarray(self.A_)                     # (K, K)

        # prob of staying in each state k (A_kk)
        stay_probs = np.diag(transmat)                     # (K,)

        # Expected duration in state k for a geom distribution
        # success: P(1 - stay_prob).
        eps = 1e-8
        expected_durations = 1.0 / np.maximum(1.0 - stay_probs, eps)

        summary = {
            "means": means,
            "vars": vars_,
            "stds": stds,
            "transmat": transmat,
            "stay_probs": stay_probs,
            "expected_durations": expected_durations,
        }
        return summary

from typing import Optional

import torch as th
from torch import nn, Tensor
from bdbf.utils import chol_inv


class Prior(nn.Module):
    def __init__(self):
        super().__init__()

    def is_valid(self):
        return False

    def reset(self):
        pass

    def info(self):
        raise NotImplementedError("Cannot get from NullPrior")


class NormalPrior(Prior):
    def __init__(self,
                 state_dim: int,
                 track: bool = False,
                 enable: bool = False):
        super().__init__()
        self.state_dim = state_dim

        self.register_buffer("mu", th.zeros(state_dim, 1))  # info vec
        self.register_buffer("sigma", th.zeros(state_dim,
                                               state_dim))  # info matrix
        self.register_buffer("n", th.tensor(0, dtype=th.long))

        self.omega = None  # information matrix
        self.xi = None  # invermation vector
        self.dirty = True

        self.track = track
        self.enable = enable

    def reset(self):
        self.mu.zero_()
        self.sigma.zero_()
        self.n.zero_()

    def is_valid(self):
        return self.n.item() > 1

    def update(self, w: Tensor):
        """Update prior with w"""
        # one pass mean and covariance computation
        i = self.n
        diff = w - self.mu
        diff_norm = diff / (i + 1)
        self.mu += diff_norm
        self.sigma += diff @ diff_norm.t() * i
        self.n += 1
        self.dirty = True

    def add(self, A: Tensor, b: Tensor, alpha: float = 1.0):
        assert not self.training

        if not self.enable:
            return A, b

        omega, xi = self.info()
        return A + omega * alpha, b + xi * alpha

    def info(self):
        assert self.is_valid()
        assert not self.training

        if self.dirty:
            sigma = self.sigma / (self.n - 1)
            omega = chol_inv(sigma)
            xi = omega @ self.mu

            self.omega = omega
            self.xi = xi
            self.dirty = False

        return self.omega, self.xi
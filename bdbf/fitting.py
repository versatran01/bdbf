from typing import List, Dict
from math import sqrt, fabs

import torch as th
from torch import nn, Tensor
from einops import rearrange

from bdbf.utils import (chol_inv, recon3d, solve_linear, masked_select_flatten,
                        th_homogenize)


def pred_var(S: Tensor, B: Tensor, eps: float = 1e-1) -> Tensor:
    """
    :param S: info matrix
    :param B: dbf
    :param eps:
    """
    m, h, w = B.shape
    # flatten dbf
    bt = rearrange(B, "m h w -> m (h w)")  # (m,n)

    try:
        # S_inv = chol_inv(S)
        S_inv = chol_inv(S.cpu()).cuda()
    except RuntimeError:
        print("Cholesky Singular")
        # Fix S and compute again
        S = S + th.diag(S.diagonal()) * eps
        # S = S + eye_like(S, eps)
        S_inv = chol_inv(S)

    var = ((S_inv @ bt) * bt).sum(0)

    var = rearrange(var, "(k h w) -> k h w", h=h, w=w)  # (1,h,w)
    return var


class LogitFit(nn.Module):
    def __init__(self, prior, in_channels: List[int], variance: bool = False):
        """
        Fit a set of bases to inverse depth
        :param in_channels: number of input channels per scale
        :param variance: whether to output variance of fitting
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 1

        self.prior = prior
        self.variance = variance

    def forward(self, bases, targets):
        """
        :param bases: [(b,m,h,w)...], list of dbf or a single set of dbf
        :param targets: (b,1,h,w), target log depth, assumes one directly take 
        the log of sparse depth (invalid pixel with 0 will be -inf, and ignored 
        during fitting)
        :return: logits, logits_var, weights. logits is just predicted log depth
        """
        # prepare input data
        assert bases[0].ndim == 4, bases[0].shape

        bases = th.cat(bases, dim=1)  # (b,m,h,w)
        # prepend bias to bases such that bases[0] = 1
        bases = th_homogenize(bases, append=False)  # (b,m,h,w)
        return self.fit_batch(bases, targets)

    def fit_batch(self, bases: Tensor, targets: Tensor):
        """
        :param bases: (b,m,h,w)
        :param targets: (b,1|2,h,w)
        :return: weights (b,m,k)
        """
        logits = []
        logits_var = []
        weights = []

        for basis, target in zip(bases, targets):
            logit, weight, logit_var = self.fit_single(basis, target)
            logits.append(logit)
            weights.append(weight)
            logits_var.append(logit_var)

        logits = th.stack(logits)  # (b,1,h,w)
        weights = th.stack(weights)  # (b,m,1)
        out = {"logit_pred": logits, "weight": weights}

        if self.variance:
            out["logit_var"] = th.stack(logits_var)

        return out

    def fit_single(self, basis: Tensor, target: Tensor):
        result = self.build_and_solve(basis, target)
        weight = result["w"]
        # A * w = b
        # (X^t P^-1 X + B^-1) * w = (X^t P^-1 y + B^-1 a)
        # use cholesky
        logit = recon3d(basis, weight)  # (k,h,w)

        logit_var = None
        if self.variance:
            # S = B @ A^-1 @ B^t
            logit_var = pred_var(result["A"], basis)

        return logit, weight, logit_var

    def check_system(self, num_weights: int, num_samples: int):
        # check system
        if self.training:
            # at train time, we must have a well-behaved system
            assert num_samples >= num_weights, \
                f"under-constrained ({num_samples} < {num_weights}) at train"
        else:
            # at eval time, unless we have a valid prior, we must also have a well-behaved system
            if not self.prior.is_valid():
                assert num_samples >= num_weights, \
                    f"under-constrained ({num_samples} < {num_weights}) at eval w/o a valid prior"

    def build_and_solve(self, basis: Tensor,
                        target: Tensor) -> Dict[str, Tensor]:
        """Build the linear system to solve
        :param basis: (m,h,w)
        :param target: (1|2,h,w), target and sqrt info, assumes diagonal covariance
        """
        assert basis.ndim == 3, basis.shape
        assert target.ndim == 3, target.shape
        assert basis.shape[-2:] == target.shape[-2:]

        logit = target[[0]]  # (1,h,w)
        mask = th.isfinite(logit)  # (1,h,w) valid mask

        M = basis.shape[0]
        N = mask.sum().item()
        self.check_system(M, N)

        if N == 0:
            # no data given, just use prior
            assert not self.training
            A, _ = self.prior.info()
            w = self.prior.mu
            return dict(A=A, w=w)

        # extract valid points
        Xt = masked_select_flatten(basis, mask)  # (m,n)
        yt = masked_select_flatten(logit, mask)  # (1,n)

        if target.shape[0] == 2:
            # sqrt info
            s = masked_select_flatten(target[[1]], mask)  # (1,n)
            Xt = Xt * s
            yt = yt * s

        XtX = Xt @ Xt.t()  # (m,n) * (n,m) = (m,m)
        Xty = Xt @ yt.t()  # (m,n) * (n,1) = (m,1)

        # update system with prior
        # A w = b
        # (X^t @ X + B^-1) w = (X^t @ y + B^-1 @ a)
        if self.training:
            # at training, assume good system so just update prior w
            w = solve_linear(XtX, Xty, lu=True)  # (m,1)
            self.prior.update(w.detach())

            # assume N >> M
            with th.no_grad():
                E_d = ((yt - w.t() @ Xt)**2).sum()
                beta = N / E_d
                A = XtX * beta
        else:
            beta0 = sqrt(N)
            alpha = 1.0
            beta = beta0
            for i in range(5):
                A, b = self.prior.add(XtX * beta, Xty * beta, alpha)

                # move some expensive calculation to cpu
                w = solve_linear(A.cpu(), b.cpu(), lu=True)  # (m,1)
                w = w.cuda()

                E_d = ((yt - w.t() @ Xt)**2).sum()

                if self.prior.use:
                    # re-estimate beta
                    S = chol_inv(A.cpu())  # (m,m)
                    S = S.cuda()

                    Tr_d = (XtX @ S).trace()

                    # re-estimate alpha
                    dw = w - self.prior.mu  # (m,1)
                    omega, _ = self.prior.info()
                    Tr_w = (omega @ S).trace()
                    E_w = dw.t() @ (omega @ dw)
                    alpha = M / (Tr_w + E_w).squeeze().item()
                else:
                    Tr_d = M / beta

                beta = N / (E_d + Tr_d).item()

                if fabs(beta / beta0 - 1.0) < 2e-2:
                    break

                beta0 = beta

            XtXb = XtX * beta
            Xtyb = Xty * beta
            A, b = self.prior.add(XtXb, Xtyb, alpha)
            w = solve_linear(A.cpu(), b.cpu(), lu=True)  # (m,1)
            w = w.cuda()

            if self.prior.track:
                self.prior.update(w)

        return dict(A=A, w=w)

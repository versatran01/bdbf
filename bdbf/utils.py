import torch as th
from torch import Tensor

from einops import rearrange


def recon3d(x: Tensor, w: Tensor) -> Tensor:
    """
    reconstruct data from weights and bases
    Apply weights to bases
    :param x: (m,h,w)
    :param w: (m,k)
    :return: (k,h,w)
    """
    return th.einsum("mhw,mk->khw", x, w)


def th_homogenize(tensor: Tensor,
                  scale: float = 1.0,
                  append: bool = True) -> Tensor:
    """
    Add one extra constant channel to tensor
    :param tensor: th.tensor (b,c,h,w)
    :param scale: scale factor
    :param append: if True add after tensor
    :return: th.tensor (b,c+1,h,w)
    """
    assert tensor.ndim >= 3, tensor.shape

    ones = th.ones_like(tensor[..., [0], :, :])  # (b,1,h,w)

    if scale != 1.0:
        ones *= scale

    if append:
        return th.cat((tensor, ones), dim=-3)  # (b,c+1,h,w)
    else:
        return th.cat((ones, tensor), dim=-3)  # (b,c+1,h,w)


def chol_inv(x: Tensor) -> Tensor:
    """Cholesky inversion"""
    return th.cholesky_inverse(th.cholesky(x))


def solve_linear(A: Tensor, b: Tensor, lu: bool = True) -> Tensor:
    """
    Solve Ax = b
    :param A: A (m,m)
    :param b: b (m,k)
    :param lu: use lu factorization, default
    :return: (m,k)
    """

    if lu:
        # x = th.lu_solve(b, *th.lu(A))
        x = th.solve(b, A)[0]
    else:
        x = th.cholesky_solve(b, th.cholesky(A))

    return x  # (m,k)


def masked_select_flatten(x: Tensor, mask: Tensor) -> Tensor:
    """Select tensor by mask and flatten"""
    return x.masked_select(mask).view(x.shape[0], -1)  # (1,N)


def calc_var_simple(info_mat, basis):
    h, w = basis.shape[-2:]
    bt = rearrange(basis, "m h w -> m (h w)")  # (m,n)

    covar = info_mat.inverse()
    covar1 = bt.t() @ covar @ bt
    var = covar1.diagonal()
    var = rearrange(var, "(k h w) -> k h w", h=h, w=w)  # (1,h,w)
    return var
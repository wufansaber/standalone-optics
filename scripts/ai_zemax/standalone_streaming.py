"""Stable, bounded-memory QR compression of affine residual blocks."""
import numpy as np


class StreamingQR:
    """Accumulate R for [J | residual], preserving its complete affine norm.

    Each input block has shape (..., rays, parameters+1). Leading axes identify
    independent fields/wavelengths/scenarios. Integration weights must already
    be applied. A block contributes rays to the SAME metric, never extra fields.
    """
    def __init__(self, columns):
        if type(columns) is not int or columns < 1:
            raise ValueError('columns must be a positive integer')
        self.columns, self.r, self.rows, self.blocks = columns, None, 0, 0
        self.peak_input_rows = 0

    def update(self, block):
        a = np.asarray(block, dtype=float)
        if a.ndim < 2 or a.shape[-1] != self.columns or a.shape[-2] < 1 or not np.isfinite(a).all():
            raise ValueError('Require nonempty finite blocks with consistent column count')
        if self.r is not None and a.shape[:-2] != self.r.shape[:-2]:
            raise ValueError('Metric leading axes changed during streaming QR')
        current = a if self.r is None else np.concatenate((self.r, a), axis=-2)
        self.r = np.linalg.qr(current, mode='r')
        self.rows += a.shape[-2]; self.blocks += 1
        self.peak_input_rows = max(self.peak_input_rows, current.shape[-2])
        return self

    def result(self):
        if self.r is None:
            raise ValueError('No residual blocks were supplied')
        return self.r.copy()


def compress_affine(jacobian, residual, *, block_rows=512):
    j, v = np.asarray(jacobian, dtype=float), np.asarray(residual, dtype=float)
    if (j.shape[:-1] != v.shape or j.ndim < 2 or type(block_rows) is not int or block_rows < 1
            or not np.isfinite(j).all() or not np.isfinite(v).all()):
        raise ValueError('Use matching finite residual/Jacobian shapes and a positive block_rows')
    qr = StreamingQR(j.shape[-1]+1)
    for begin in range(0, j.shape[-2], block_rows):
        qr.update(np.concatenate((j[..., begin:begin+block_rows, :], v[..., begin:begin+block_rows, None]), axis=-1))
    return qr.result()


def validate_compressed(value, base_residual, parameters):
    """Check external compressed blocks before a solver trusts their shape/norm.

    This checks the current residual norm, not the supplied derivative itself;
    directional finite differences are still required for every optical adapter.
    """
    r, v = np.asarray(value, dtype=float), np.asarray(base_residual, dtype=float)
    if (r.ndim != v.ndim+1 or r.shape[:-2] != v.shape[:-1] or r.shape[-1] != parameters+1
            or not 1 <= r.shape[-2] <= parameters+1 or not np.isfinite(r).all()
            or not np.allclose(np.linalg.norm(r[..., -1], axis=-1), np.linalg.norm(v, axis=-1), rtol=2e-9, atol=1e-11)):
        raise ValueError('Compressed affine block does not match current residual axes/norm')
    return r.copy()


def affine_rank_factor(matrix,relative_tolerance):
    """SVD rows for a small affine block; retain a certified tail spectral norm."""
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all()or not np.isfinite(relative_tolerance)or relative_tolerance<0.:
        raise ValueError('Require a finite affine matrix and nonnegative rank tolerance')
    if relative_tolerance==0.:return a.copy(),0.
    _,s,vt=np.linalg.svd(a,full_matrices=False)
    if not len(s):raise ValueError('Affine norm matrix cannot be empty')
    rank=int(np.sum(s>relative_tolerance*s[0]))if s[0]else 0
    if rank==len(s):return a.copy(),0.
    return s[:rank,None]*vt[:rank],float(s[rank])


def bounded_affine_rank(factor,tail_spectral_norm,lower,upper):
    """Conservative smaller norm valid throughout the FULL physical trust box.

    Orthogonal SVD rows give ||A[d;1]||²=||A_kept[d;1]||²+||A_tail[d;1]||².
    The omitted tail is bounded by sigma_tail*sqrt(1+sum(max|d_i|²)). Appending
    that constant as one row gives an UPPER norm, never an optimistic truncation.
    """
    factor=np.asarray(factor,dtype=float);lo,hi=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if factor.ndim!=2 or lo.ndim!=1 or lo.shape!=hi.shape or factor.shape[-1]!=len(lo)+1 or not np.isfinite(factor).all()or np.any(lo>hi)or not np.isfinite(lo).all()or not np.isfinite(hi).all()or not np.isfinite(tail_spectral_norm)or tail_spectral_norm<0.:
        raise ValueError('Invalid affine rank bound/trust box')
    if tail_spectral_norm==0.:return factor.copy()
    bound=tail_spectral_norm*np.sqrt(1+np.sum(np.maximum(abs(lo),abs(hi))**2))
    extra=np.zeros((1,factor.shape[-1]));extra[0,-1]=bound
    return np.r_[factor,extra]

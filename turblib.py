"""Helper functions to make power spectrum from 2d fields."""

__all__ = [
    "calc_spec2d",
    "plot_spec",
]

import numpy as np

HAVE_SCIPY = False
try:
    import scipy.fft

    HAVE_SCIPY = True
except:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt


def calc_spec2d(f, dx=1, dy=1, bins=None, k_2d=None, use_scipy_fft=False):
    """Compute spectrum of abs(FFT(f))**2 vs |k|.

    TODOs:
    - Use in-place FFT in scipy to save memory.

    Args:
    f: 2d ndarray of shape (ny, nx).
    dx, dy: Numbers.
    bins: 1d ndarray of the histogram bin edges or None. If None, the positive
        half of kx will be used.
    k_2d: 2d ndarray of the 2d |k| array or None. If None, one will be computed
        from the k coordinates. Otherwise, one may supply a pre-computed copy
        to avoid repeated calculation.

    Returns:
    k_2d: 2d ndarray of the 2d |k| array.
    hist: 1d ndarray of the binned histogram.
    bin_edges: 1d ndarray of the histogram bin edges.
    """
    f_ = np.squeeze(np.array(f))
    if f_.ndim != 2:
        raise ValueError(f"f must be 2d array; got {f_.ndim}d")
    ny, nx = f_.shape

    if use_scipy_fft:
        fft_f_ = scipy.fft.fftn(f_)
        fft_f = scipy.fft.fftshift(fft_f_)
    else:
        fft_f_ = np.fft.fftn(f_)
        fft_f = np.fft.fftshift(fft_f_)
    power = abs(fft_f) ** 2

    if k_2d is None:
        kx_ = np.fft.fftfreq(nx, d=dx)
        ky_ = np.fft.fftfreq(ny, d=dy)
        kx = np.fft.fftshift(kx_)
        ky = np.fft.fftshift(ky_)
        kx_2d, ky_2d = np.meshgrid(kx, ky, indexing="ij")
        k_2d = np.sqrt(kx_2d ** 2 + ky_2d ** 2)

    if bins is None:
        bins_ = kx[nx // 2 :]
    else:
        bins_ = bins
    hist, bin_edges = np.histogram(k_2d, bins_, weights=power)

    return k_2d, hist, bin_edges


def plot_spec(hist, bin_edges, ax=None, style="step", **kwargs):
    if ax is None:
        ax = plt.gca()
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if style == "step":
        ax.step(bin_centers, hist, where="mid", **kwargs)
    elif style == "line":
        ax.plot(bin_centers, hist, **kwargs)
    else:
        error_msg = f'style must "step" or "line"; got {style}'
        raise ValueError(msg)
    ax.set_xscale("log")
    ax.set_yscale("log")

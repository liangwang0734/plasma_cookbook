"""Cartesian grid wrapper to allow slicing and simple gradient calculations."""

__all__ = ["UniformGrid"]


import functools
import numpy as np


def unsized2sized(array):
    array_ = np.array(array)
    if array_.ndim == 0:
        array_ = np.array_((array_,))
    return array_


class UniformGrid:
    def __init__(self, nxyz, xyzl, xyzu):
        """Create an object to handle 1d to 3d uniform grid.

        TODO: Handling degenerate dimensions.

        Args:
        nxyz: 1d: nx or (nx,); 2d: (nx, ny); 3d: (nx, ny, nz).
        xyzl: Lower bounds. 1d: xl or (xl,); 2d: (xl, yl); 3d: (xl, yl, zl).
        xyzu: Upper bounds. 1d: xu or (xu,); 2d: (xu, yu); 3d: (xu, yu, zu).
        """
        self.nxyz = unsized2sized(nxyz)
        self.xyzl = unsized2sized(xyzl)
        self.xyzu = unsized2sized(xyzu)
        self.ndim = len(self.nxyz)
        self.dxyz = (self.xyzu - self.xyzl) / self.nxyz
        self.dims = list("xyz"[: self.ndim])

        self.nx = self.nxyz[0]
        self.xl = self.xyzl[0]
        self.xh = self.xyzu[0]
        self.dx = self.dxyz[0]

        if self.ndim > 1:
            self.ny = self.nxyz[1]
            self.yl = self.xyzl[1]
            self.yh = self.xyzu[1]
            self.dy = self.dxyz[1]

            if self.ndim > 2:
                self.nz = self.nxyz[2]
                self.zl = self.xyzl[2]
                self.zh = self.xyzu[2]
                self.dz = self.dxyz[2]

        self.ndim_true = np.count_nonzero(self.nxyz - 1)

        if self.ndim_true == 3:
            self.dims_true = ["x", "y", "z"]
        elif self.ndim_true == 2 and self.ndim == 2:
            self.dims_true = ["x", "y"]
        elif self.ndim_true == 2 and self.ndim == 3:
            if self.nx == 1:
                self.dims_true = ["y", "z"]
            elif self.ny == 1:
                self.dims_true = ["x", "z"]
            else:
                self.dims_true = ["x", "y"]
        elif self.ndim_true == 1 and self.ndim == 1:
            self.dims_true = ["x"]
        elif self.ndim_true == 1 and self.ndim == 2:
            if self.nx == 1:
                self.dims_true = ["y"]
            else:
                self.dims_true = ["x"]
        elif self.ndim_true == 1 and self.ndim == 3:
            if self.nx == 1 and self.ny == 1:
                self.dims_true = ["z"]
            elif self.nx == 1 and self.nz == 1:
                self.dims_true = ["y"]
            else:
                self.dims_true = ["x"]

    def coords(self, dim=None, center="cc"):
        """Get 1d cell-center or node-center coordinates.

        Args:
        dim: One of 'x', 'y', and 'z'.
        center: One of 'cc' and 'nc' for cell-center and node-center
            coordinates, respectively.

        Returns:
        A 1d array for cell-center or node-center coordinates.
        """
        if self.ndim == 1 and dim is None:
            dim = "x"
        d = self.dims.index(dim)

        if center == "nc":
            nc = np.linspace(self.xyzl[d], self.xyzu[d], self.nxyz[d] + 1)
            return nc
        elif center == "cc":
            nc = self.coords(dim, "nc")
            cc = 0.5 * (nc[1:] + nc[:-1])
            return cc
        else:
            raise ValueError(f"center must be nc or cc. Got {center}.")

    # Compute x, y, or z coordinates only once if retreived.
    @property
    @functools.lru_cache()
    def x(self):
        """Get the 1d array for cell-centered x coordinates."""
        return self.coords("x")

    @property
    @functools.lru_cache()
    def y(self):
        """Get the 1d array for cell-centered y coordinates."""
        return self.coords("y")

    @property
    @functools.lru_cache()
    def z(self):
        """Get the 1d array for cell-centered z coordinates."""
        return self.coords("z")

    @property
    @functools.lru_cache()
    def xnc(self):
        """Get the 1d array for node-center x coordinates."""
        return self.coords("x", "nc")

    @property
    @functools.lru_cache()
    def ync(self):
        """Get the 1d array for node-center y coordinates."""
        return self.coords("y", "nc")

    @property
    @functools.lru_cache()
    def znc(self):
        """Get the 1d array for node-center z coordinates."""
        return self.coords("z", "nc")

    def ddx(self, data=None, **kwargs):
        """d/dx operator"""
        if "x" in self.dims_true:
            return np.gradient(data, self.x, axis=-1, **kwargs)
        else:
            return 0.0

    def ddy(self, data, **kwargs):
        """d/dy operator"""
        yaxis = (self.dims).index("y")
        if "y" in self.dims_true:
            return np.gradient(data, self.y, axis=yaxis, **kwargs)
        else:
            return 0.0

    def ddz(self, data, **kwargs):
        """d/dz operator"""
        if "z" in self.dims_true:
            return np.gradient(data, self.z, axis=0, **kwargs)
        else:
            return 0.0

    def curl_x(self, vx, vz, **kwargs):
        """x-component of curl(v), where v=(vy,vz) is a vector"""
        return self.ddy(vz) - self.ddz(vy)

    def curl_y(self, vx, vz, **kwargs):
        """y-component of curl(v), where v=(vx,vz) is a vector"""
        return self.ddz(vx) - self.ddx(vz)

    def curl_z(self, vx, vy, **kwargs):
        """z-component of curl(v), where v=(vx,vy) is a vector"""
        return self.ddx(vy) - self.ddy(vx)

    def div(self, v1, v2=None, v3=None, **kwargs):
        """div(v), where v=(v1,v2,v3) is a vector"""
        if self.ndim_true == 3:
            return self.ddx(v1) + self.ddz(v2) + self.ddz(v3)
        elif self.ndim_true == 2:
            if self.dims_true == ["x", "y"]:
                return self.ddx(v1) + self.ddy(v2)
            elif self.dims_true == ["x", "z"]:
                return self.ddx(v1) + self.ddz(v2)
            elif self.dims_true == ["y", "z"]:
                return self.ddy(v1) + self.ddz(v2)
            else:
                raise ValueError(f"dims_true {self.dims_true} is not supported")
        else:
            if self.dims_true == ["x"]:
                return self.ddx(v1)
            elif self.dims_true == ["y"]:
                return self.ddy(v1)
            elif self.dims_true == ["z"]:
                return self.ddz(v1)
            else:
                raise ValueError(f"dims_true {self.dims_true} is not supported")

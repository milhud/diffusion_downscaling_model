"""ERA5 regridding to CONUS404 grid using xESMF. Weights held in memory only."""

import numpy as np
import xesmf as xe
import xarray as xr


class ERA5Regridder:
    """Regrids ERA5 regular lat/lon data to the CONUS404 curvilinear grid.

    The weight matrix is computed once at init and held in memory.
    Uses bilinear interpolation with nearest-neighbor extrapolation
    for CONUS404 pixels outside ERA5 bounds (~32%).
    """

    def __init__(self, era5_lat: np.ndarray, era5_lon: np.ndarray,
                 conus_lat: np.ndarray, conus_lon: np.ndarray):
        """
        Args:
            era5_lat: (111,) latitude array (descending)
            era5_lon: (235,) longitude array
            conus_lat: (1015, 1367) 2D latitude array
            conus_lon: (1015, 1367) 2D longitude array
        """
        # Build source grid (ERA5 — rectilinear)
        self.src_grid = xr.Dataset({
            "lat": xr.DataArray(era5_lat, dims=["y"]),
            "lon": xr.DataArray(era5_lon, dims=["x"]),
        })

        # Build target grid (CONUS404 — curvilinear)
        self.dst_grid = xr.Dataset({
            "lat": xr.DataArray(conus_lat, dims=["y", "x"]),
            "lon": xr.DataArray(conus_lon, dims=["y", "x"]),
        })

        # Build regridder — bilinear with extrapolation
        self.regridder = xe.Regridder(
            self.src_grid, self.dst_grid,
            method="bilinear",
            extrap_method="nearest_s2d",
            unmapped_to_nan=False,
        )

    def regrid(self, data: np.ndarray) -> np.ndarray:
        """Regrid a single 2D field from ERA5 grid to CONUS404 grid.

        Args:
            data: (111, 235) array on ERA5 grid
        Returns:
            (1015, 1367) array on CONUS404 grid
        """
        da = xr.DataArray(data, dims=["y", "x"])
        return self.regridder(da).values

    def regrid_batch(self, data: np.ndarray) -> np.ndarray:
        """Regrid a stack of 2D fields: (C, 111, 235) → (C, 1015, 1367)."""
        out = np.stack([self.regrid(data[i]) for i in range(data.shape[0])])
        return out

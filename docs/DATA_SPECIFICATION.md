# Data Specification

## Variable Table

### Paired Variables (ERA5 input -> CONUS404 output)

| ERA5 Name | CONUS404 Name | Description | Units | Pretransform |
|-----------|---------------|-------------|-------|--------------|
| t2m | T2 | 2-meter temperature | K | none |
| d2m | TD2 | 2-meter dewpoint temperature | K | none |
| u10 | U10 | 10-meter U-wind component | m/s | none |
| v10 | V10 | 10-meter V-wind component | m/s | none |
| sp | PSFC | Surface pressure | Pa | none |
| tp | PREC_ACC_NC | Non-convective precipitation | mm | log1p |

### Synthesis Variable (output only, no direct ERA5 counterpart)

| CONUS404 Name | Description | Units | Pretransform |
|---------------|-------------|-------|--------------|
| Q2 | 2-meter specific humidity | kg/kg | sqrt |

Q2 is predicted without a direct ERA5 input variable. The model learns Q2 from correlated inputs (temperature, dewpoint, pressure, moisture-related static fields). This mirrors CorrDiff's radar reflectivity synthesis.

### Static Fields (input only, 6 channels)

| Channel | Name | Description | Source |
|---------|------|-------------|--------|
| 0 | Terrain | Surface elevation | CONUS404 Z level 0 |
| 1 | Orog. variance | Subgrid orographic variability | Computed from terrain |
| 2 | Latitude | Grid latitude | CONUS404 coordinate |
| 3 | Longitude | Grid longitude | CONUS404 coordinate |
| 4 | LAI | Leaf area index | ERA5, regridded |
| 5 | Land-sea mask | Land (1) vs ocean (0) | ERA5, regridded |

## Grid Specifications

### ERA5 (Input)
- **Type:** Regular latitude-longitude
- **Resolution:** 0.25 degrees (~27 km)
- **Dimensions:** 111 (lat) x 235 (lon)
- **Coverage:** Global subset covering CONUS

### CONUS404 (Target)
- **Type:** WRF Lambert conformal conic projection
- **Resolution:** ~4 km
- **Dimensions:** 1015 (south_north) x 1367 (west_east)
- **Coverage:** Contiguous United States

### Patch Size
- **Training patches:** 256 x 256 pixels (~1024 km x 1024 km at 4 km resolution)
- **Valid patch origins:** 87 (patches with >= 80% land fraction)
- **Patches per day:** 2 (randomly selected from valid origins)

## Channel Counts

```
IN_CH  = len(ERA5_VARS) + NUM_STATIC_FIELDS
       = 6 + 6 = 12

OUT_CH = len(CONUS404_VARS)  # including synthesis vars
       = 7

LATENT_CH = 8                # VAE latent channels (when OUT_CH > 1)
```

## Data Split

| Split | Years | Purpose |
|-------|-------|---------|
| Train | 1980-2014 (35 years) | Model training |
| Validation | 2015-2017 (3 years) | Hyperparameter tuning, early stopping |
| Test | 2018-2020 (3 years) | Final evaluation, paper results |

## Data Sources

- **ERA5:** ECMWF Reanalysis v5
  - Location: `/discover/nobackup/sduan/pipeline/data/processed/era5_{YYYY}.nc`
  - Symlinked from: `data/era5_{YYYY}.nc`

- **CONUS404:** WRF dynamical downscaling of ERA5 by NCAR
  - Location: `/discover/nobackup/hpmille1/final_data/conus404_yearly_{YYYY}.nc`
  - Symlinked from: `data/conus404_yearly_{YYYY}.nc`

## Available Variables in Source Files

### ERA5 (all available in .nc files)
`cvh, cvl, d2m, lai_hv, lai_lv, lsm, sp, t2m, tp, u10, v10, z, overlap, lai`

### CONUS404 (all available in .nc files)
`T2, Q2, TD2, PSFC, ACRAINLSM, LAI, U10, V10, Z, W, PREC_ACC_NC`

## Known Data Issues

1. **63 NaN days:** Years 2004 (28d), 2005 (32d), 2008 (1d), 2009 (2d) have fully-NaN CONUS404 data. Tracked in `cached_data/nan_days.json`.

2. **ERA5 time structure:** The (12, 366) leading dims require month-day lookup since each month only populates its own day indices.

3. **Grid mismatch:** ~32% of CONUS404 pixels fall outside ERA5 lat/lon bounds, handled by nearest-neighbor extrapolation during regridding.

4. **LAI has 452 NaN pixels:** Filled with 0.0 during preprocessing.

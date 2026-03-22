"""Model configuration — variables, bounds, and training defaults.

Add new variable pairs to VARIABLE_PAIRS to train on more fields.
Currently set to temperature only.
"""

# ─── US CONUS geographic bounds ──────────────────────────────────────────────
LAT_MIN, LAT_MAX = 24, 50
LON_MIN, LON_MAX = -125, -66

# ─── Variable pairs: ERA5 name → CONUS404 name ──────────────────────────────
# Only temperature for now — uncomment lines below to add more variables.
VARIABLE_PAIRS = {
    "t2m": "T2",
    "d2m": "TD2",
    "u10": "U10",
    "v10": "V10",
    "sp": "PSFC",
    "tp": "PREC_ACC_NC",
    # NOTE: "z" (geopotential) has no CONUS404 equivalent — removed.
    # Q2 (specific humidity) has no direct ERA5 single-level match either.
}

ERA5_VARS = list(VARIABLE_PAIRS.keys())
CONUS404_VARS = list(VARIABLE_PAIRS.values())

# Number of input channels = len(ERA5_VARS) + num_static_fields
NUM_STATIC_FIELDS = 6  # terrain, orog_var, lat, lon, lai, lsm
IN_CH = len(ERA5_VARS) + NUM_STATIC_FIELDS
OUT_CH = len(CONUS404_VARS)

# ─── Units and labels ────────────────────────────────────────────────────────
VARIABLE_UNITS = {
    "t2m": "K", "d2m": "K", "sp": "hPa",
    "u10": "m/s", "v10": "m/s", "tp": "mm", "z": "m",
    "T2": "K", "TD2": "K", "PSFC": "hPa",
    "U10": "m/s", "V10": "m/s", "PREC_ACC_NC": "mm", "Q2": "kg/kg",
}

VARIABLE_NAMES = {
    "t2m": "2m Temperature", "d2m": "2m Dewpoint", "sp": "Surface Pressure",
    "u10": "10m U-Wind", "v10": "10m V-Wind", "tp": "Precipitation",
    "z": "Geopotential Height",
    "T2": "2m Temperature", "TD2": "2m Dewpoint", "PSFC": "Surface Pressure",
    "U10": "10m U-Wind", "V10": "10m V-Wind", "PREC_ACC_NC": "Precipitation",
    "Q2": "Specific Humidity",
}

# ─── Pretransforms (applied before z-scoring) ────────────────────────────────
PRETRANSFORMS = {
    "tp": "log1p",
    "PREC_ACC_NC": "log1p",
    "Q2": "sqrt",
    "z": "geopotential",
}

# ─── Model architecture ─────────────────────────────────────────────────────
PATCH_SIZE = 256
LATENT_H = LATENT_W = 64
LATENT_CH = 8 if OUT_CH > 1 else 4

MODEL = dict(
    drn_base_ch=64,
    drn_ch_mults=(1, 2, 4, 8),
    drn_num_res_blocks=2,
    drn_attn_resolutions=(2,),
    vae_base_ch=128,
    diff_base_ch=128,
    diff_ch_mults=(1, 2, 2, 4),
    diff_num_res_blocks=4,
    diff_attn_resolutions=(1, 2, 3),
    diff_time_dim=512,
)

# ─── Training defaults ──────────────────────────────────────────────────────
TRAIN = dict(
    train_years=list(range(1980, 2015)),
    val_years=list(range(2015, 2018)),
    test_years=list(range(2018, 2021)),
    batch_size=8,
    patches_per_day=2,
    drn_epochs=50,
    drn_lr=2e-4,
    drn_warmup_epochs=5,
    vae_epochs=25,
    vae_lr=1e-4,
    vae_warmup_epochs=3,
    vae_beta_max=1e-3,
    vae_beta_anneal_frac=0.3,
    diff_epochs=50,
    diff_lr=2e-4,
    diff_warmup_epochs=5,
    diff_grad_accum=4,        # accumulate 4 mini-batches → effective batch 32
    diff_cosine_restart_period=10,  # restart LR every N epochs after warmup
    diff_p_mean=-0.8,         # EDM noise schedule: shifted toward lower noise
    diff_p_std=1.0,           # EDM noise schedule: tighter distribution
    ema_decay=0.9999,
    p_uncond=0.1,
    min_land_frac=0.8,  # patches must be >= 80% land
)

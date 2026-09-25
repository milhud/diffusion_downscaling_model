# Literature notes: what Mardani (CorrDiff) and Lopez-Gomez (R2-D2) evaluate

Sources: arXiv HTML of Mardani et al. (2309.15214) fetched this session; R2-D2 (arXiv 2410.01776 / PNAS 122(17), 2025)
from search-result abstracts and the repo's own `docs/PUBLISHABLE_POINTS.md` / `docs/paper_points.md` only
(the R2-D2 full text was too large to fetch, so anything R2-D2 beyond the abstract is marked **[unverified]**).
No local PDF of either paper exists in this repository.

## Mardani et al. 2025 — CorrDiff (Communications Earth & Environment 6:124)

| Item | What the paper does |
|---|---|
| Decomposition | x = μ + r. UNet regression predicts the conditional mean μ; a diffusion model generates the residual r. |
| Why "corrective" | Direct conditional diffusion "showed slow convergence and poor-quality images with incoherent structures". The regression step removes the large-scale, deterministic part (topography effects, large-scale winds, i.e. the systematic coarse-model error), reducing target variance so diffusion only models the fine-scale stochastic remainder. |
| Domain / data | Taiwan, ERA5 25 km (36x36) -> WRF 2 km (448x448); 12 in-channels, 4 out (T2, u10, v10, radar reflectivity); train 2018-2020 hourly (~24k samples), test 2021 (205 random dates). |
| Baselines | ERA5 interpolation, UNet (regression only), Random Forest. |
| Reported numbers | Radar CRPS 1.90 (CorrDiff) vs 2.51 (UNet) vs 3.56 (RF). t2m MAE 0.65 / 0.64 / 0.81 / 0.97 (CorrDiff / UNet / RF / ERA5 interp); u10m MAE 1.08 / 1.10 / 1.14 / 1.17. |
| Events | **Typhoon Haikui (2023-09-03 00 UTC)**: max 10 m wind 22 (ERA5) -> 33 (CorrDiff) vs 45 m/s (WRF); radius of max wind 75 -> 50 km; wind PDF tail restored to 40 m/s (ERA5 capped 27). **Typhoon Chanthu (2021)** appendix: CorrDiff over-contracts the cyclone. Cold-front case: sharpens wind/temperature gradients co-located with intense rain. |
| Metrics | MAE, CRPS, radially-averaged power spectra (10-200 km kinetic energy), PDFs incl. tails, spread-skill (found **under-dispersive**). |
| Stated limitation | "Beyond the coherence of the large scale conditioning given from ERA5 there is no guarantee that CorrDiff's km-scale dynamics will be coherent in time" — each frame is sampled independently; temporal coherence (video diffusion / autoregression) is left as future work. |

**Bias-correction point (as raised in the brief):** the first phase (regression) is the coarse-model *correction*; it is what makes
this a bias-correcting downscaler rather than pure super-resolution. Our `bias_decomposition.png` tests exactly this on our model.

## Lopez-Gomez et al. 2025 — R2-D2 (PNAS 122(17))

| Item | Detail |
|---|---|
| Design | ESM output -> WRF dynamical downscaling to 45 km (homogenises different ESMs) -> diffusion refinement to 9 km. Residual between 9 km and 45 km fields is what the diffusion model samples. |
| Cost claim | The 45 -> 9 km dynamical stage is ~40x more expensive and >97.5% of the original system's cost; >800 samples/hour on 16 A100s. |
| Metric | CRPS (equals MAE in deterministic limit); CRPS reduced by >40% vs bias-corrected spatial disaggregation (BCSD). |
| Events (per repo notes) | Compound extremes / Santa Ana wind case studies; multi-ESM generalisation (trained on one CMIP6 model, tested on others). |
| Temporal treatment | **[unverified]** not confirmed from the fetched sources. |

## Can we benchmark against the two models? (checked)

* **Mardani CorrDiff weights:** none on this system (searched `/gpfsm/dnb33/hpmille1`, `/discover/nobackup/hpmille1` for corrdiff / physicsnemo / modulus / earth2). A public **Taiwan** checkpoint exists (NVIDIA NGC "CorrDiff inference package", Apache-2.0; `CorrDiffTaiwan` in Earth2Studio: inputs tcwv, z/t/u/v 500 & 850 hPa, t2m, u10m, v10m; outputs mrr, t2m, u10m, v10m; trained on ERA5→WRF 2018-2021 hourly) but it is Taiwan-specific (448x448 grid, pressure-level inputs we do not hold), so it cannot be run on CONUS ERA5 without retraining. Instead, **CorrDiff-style and R2-D2-style re-implementations were trained on our data** and benchmarked (README §10).
* **Lopez R2-D2 weights/code:** none available locally; its first stage is a WRF simulation we cannot run.
* **Local stand-ins actually available:**
  1. **This repo's DRN** = CorrDiff's regression stage; **DRN + latent diffusion** = CorrDiff with latent-space diffusion (our model). So the "CorrDiff-style" comparison is DRN-vs-ensemble.
  2. **`/gpfsm/dnb33/hpmille1/model` (MRPD)** — a T2-only 25->12->4 km cascade with a regression + diffusion model at each stage: the closest local analogue to R2-D2's multi-stage design. Caveats: T2 only, 900 km domain, trained 2013-15/2017-18 (so 2018 events are in its training years), only its regression cascade is runnable (no sampler in the repo).
     **Result of running it (see README §10): the checkpoints are non-functional** — on MRPD's own training-domain centre crop its regression cascade has 7-13 K RMSE vs 1.9 K for plain ERA5 interpolation; stage-2 loss plateaued near 0.51, the stage-3 history is empty and the stage-3 checkpoints contain all-NaN `stage2_*` weights. It is therefore not a usable benchmark.
  3. `src/evaluation/ablation_pixel.py` (pixel-space CorrDiff ablation) exists as code but **no trained checkpoint** exists in `checkpoints/`; only timing numbers (`results/benchmark`) are available.
* Literature numbers above are context only: different domain, resolution and variables, so they are **not** directly comparable to our numbers.

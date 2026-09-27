"""Sanity check: does MRPD work on ITS OWN training domain (centre 225 px crop)?"""
import sys
sys.path.insert(0, "/gpfsm/dnb33/hpmille1/model")
import numpy as np, torch, torch.nn.functional as F
import train as mrpd
from config import DataConfig
sys.path.insert(0, "/gpfsm/dnb33/hpmille1/diffusion_downscaling_model/scripts")
import mrpd_event_baseline as B
C = "/discover/nobackup/sduan/.data"; T = 225; MEAN, STD = DataConfig.TEMP_MEAN, DataConfig.TEMP_STD
dev = "cuda"
model = mrpd.MRPDModel().to(dev).eval()
B.load(model, 1, 15); B.load(model, 2, 15); B.load(model, 3, 20)
for year in (2019, 2020):
    era = np.load(f"{C}/era5_{year}.npy", mmap_mode="r"); con = np.load(f"{C}/conus_{year}.npy", mmap_mode="r")
    H, W = era.shape[-2:]; y0, x0 = H // 2 - T // 2, W // 2 - T // 2
    out = {"era5": [], "s1": [], "s2": [], "s3": []}
    for d in range(0, 360, 30):
        e = np.asarray(era[d, 0, y0:y0 + T, x0:x0 + T]); t = np.asarray(con[d, 0, y0:y0 + T, x0:x0 + T])
        x = torch.from_numpy(((e - MEAN) / STD)[None, None].astype(np.float32)).to(dev)
        with torch.no_grad():
            s1, _ = model.forward_stage1(F.adaptive_avg_pool2d(x, 36)); m2, _ = model.forward_stage2(s1); m3, _ = model.forward_stage3(m2)
        rm = lambda p: float(np.sqrt(np.mean((p - t) ** 2))); 
        up1 = F.interpolate(s1, size=T, mode="bilinear").cpu().numpy()[0, 0] * STD + MEAN
        out["era5"].append(rm(e)); out["s1"].append(rm(up1))
        out["s2"].append(rm(m2[0, 0].cpu().numpy() * STD + MEAN)); out["s3"].append(rm(m3[0, 0].cpu().numpy() * STD + MEAN))
    print(year, {k: round(float(np.mean(v)), 2) for k, v in out.items()}, "(RMSE K vs CONUS404 on MRPD's own centre crop, 12 days)", flush=True)

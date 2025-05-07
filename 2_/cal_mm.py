from klimtool import klimtool # main module

import xarray as xr
import datetime
import geopandas as gpd
import rioxarray
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from windrose import WindroseAxes

klimtool = klimtool()
timestart = datetime.datetime(2020,1,1)
timeend   = datetime.datetime(2025,5,1)
lonw = 90
lone = 145
latb = -15
latt = 15
latlon = [lonw, lone, latb, latt]
import time
from datetime import datetime
from dask.diagnostics import ProgressBar
from contextlib import redirect_stdout
import sys
from pathlib import Path
import shutil
import dask

# Atur log file
log_file_path = "export_log.txt"
log = open(log_file_path, "a")  # mode append agar tidak menimpa

def logprint(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")
    print(f"[{now}] {msg}", file=log)

logprint("🚀 Starting export process...")

def safe_rmdir(path):
    p = Path(path)
    if p.exists() and p.is_dir():
        logprint(f"🗑️ Removing existing directory: {path}")
        shutil.rmtree(p)
    else:
        logprint(f"✅ No need to remove {path}, does not exist.")
zarr_waves_path = "/data/local/marine-training/data/ofs/inawaves_monthly_mean.zarr"
zarr_flows_path = "/data/local/marine-training/data/ofs/inaflows_monthly_mean.zarr"
safe_rmdir(zarr_waves_path)
safe_rmdir(zarr_flows_path)

# Open datasets
logprint("📥 Opening inawaves dataset...")
dsinawaves = klimtool.open_inawaves(tstart=timestart, tend=timeend, latlon=latlon)

logprint("📥 Opening inaflows dataset...")
dsinaflows = klimtool.open_inaflows(tstart=timestart, tend=timeend, latlon=latlon)

# Export waves with progress bar (to log and console)
logprint("💾 Exporting waves to Zarr...")
start_time = time.time()
with dask.config.set(scheduler='threads'):
    with ProgressBar():
        dsrose = dsinawaves.resample(time='MS').mean()
        dsrose.to_zarr("/data/local/marine-training/data/ofs/inawaves_monthly_mean.zarr")
duration = time.time() - start_time
logprint(f"✅ Waves export done in {duration:.2f} seconds.")

# Export flows with progress bar (to log and console)
logprint("💾 Exporting flows to Zarr...")
start_time = time.time()
with dask.config.set(scheduler='threads'):
    with ProgressBar():
        dsroseflow = dsinaflows.resample(time='MS').mean()
        dsroseflow.to_zarr("/data/local/marine-training/data/ofs/inaflows_monthly_mean.zarr")
duration = time.time() - start_time
logprint(f"✅ Flows export done in {duration:.2f} seconds.")

logprint("🎉 Export process completed.\n")
log.close()

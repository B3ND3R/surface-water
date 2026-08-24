# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SWIM Water Extent over Kenya — animated time series
#
# This notebook pulls the **SWIM-WE (Surface Water Inventory and Monitoring — Water Extent)** dataset from DLR/EOC and animates how surface water extent changes over time for an area of interest in Kenya.
#
# - Dataset page: https://geoservice.dlr.de/web/datasets/swim
# - SWIM-WE is a daily, 10 m resolution, binary water/non-water classification derived from individual Sentinel-1 and Sentinel-2 scenes using deep neural networks.
# - Data is served as per-scene cloud-optimized GeoTIFFs (COGs), discoverable through a STAC API. There is **no single pre-made "Kenya" mosaic** — each STAC item covers one satellite tile on one date, so we search for tiles that intersect our area of interest (AOI), clip each one to the AOI, and mosaic same-day tiles together to build a time series.
# - License: CC-BY-NC-4.0 (non-commercial use; cite DLR/EOC — see references in the collection metadata).
#
# **Why a lake, not the whole country:** Kenya spans dozens of Sentinel-2 tiles, each imaged on a different day. Building a literal daily wall-to-wall country mosaic at 10 m would mean downloading and stitching a huge number of COGs with inconsistent dates per pixel — not a great fit for a clean animated time series. Instead, this notebook focuses by default on **Lake Naivasha**, a Rift Valley lake known for large water-level swings, which fits inside a single Sentinel-2 tile and produces a meaningful, fast-to-build animation. The `AOI_BBOX` below is just a variable — widen it to any other Kenyan water body (Lake Turkana, Lake Nakuru, Tana River delta, etc.) or, with patience, the whole country (see the note near the config cell).

# %%
# Install dependencies (safe to re-run).
# %pip install -q pystac-client rasterio numpy matplotlib pillow pandas

# %%
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from pystac_client import Client

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from IPython.display import Image as IPyImage

# %% [markdown]
# ## Configuration
#
# `AOI_BBOX` is `(west, south, east, north)` in degrees. Some other Kenyan water bodies you can swap in:
#
# - Lake Naivasha (default): `(36.15, -0.85, 36.45, -0.65)`
# - Lake Nakuru: `(36.00, -0.42, 36.15, -0.28)`
# - Lake Turkana (larger AOI, slower): `(35.9, 2.3, 36.8, 4.7)`
# - Whole of Kenya (many tiles, slow — expect a long run and a large output grid): `(33.9, -4.7, 41.9, 5.0)`
#
# `MIN_ITEM_VALID_PCT` filters out scenes that are mostly cloud-covered *for the whole satellite tile*. `MIN_AOI_COVERAGE` then filters out dates where, after clipping to the AOI and mosaicking same-day tiles, less than that fraction of the AOI actually has a usable (non-cloud, non-nodata) classification — this matters because a tile can be mostly clear overall but cloudy exactly over our AOI.

# %%
STAC_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION = "SWIM_WE"

AOI_NAME = "Lake Naivasha, Kenya"
AOI_BBOX = (36.15, -0.85, 36.45, -0.65)  # (west, south, east, north)

# Optional ISO date filters, e.g. "2024-01-01". Leave as None to use everything available.
DATE_START = None
DATE_END = None

RESOLUTION_DEG = 0.0001  # ~10 m at the equator, matches native SWIM-WE resolution
MIN_ITEM_VALID_PCT = 60  # STAC item-level 'valid' statistic (% of tile that isn't cloud/nodata)
MIN_AOI_COVERAGE = 0.5   # required fraction of AOI pixels classified after mosaicking

CACHE_FILE = Path(f"swim_cube_{AOI_NAME.split(',')[0].replace(' ', '_').lower()}.npz")

# %% [markdown]
# ## Search the STAC API for tiles covering the AOI

# %%
client = Client.open(STAC_URL)

search_kwargs = dict(collections=[COLLECTION], bbox=AOI_BBOX, limit=500)
if DATE_START or DATE_END:
    search_kwargs["datetime"] = f"{DATE_START or '..'}/{DATE_END or '..'}"

items = list(client.search(**search_kwargs).items())
print(f"Found {len(items)} scenes intersecting the AOI")

items = [it for it in items if it.properties.get("statistics", {}).get("valid", 0) >= MIN_ITEM_VALID_PCT]
items.sort(key=lambda it: it.datetime)
print(f"{len(items)} scenes remain after the tile-level valid-pixel filter")

by_date = defaultdict(list)
for it in items:
    by_date[it.datetime.date()].append(it)
print(f"{len(by_date)} unique acquisition dates, {min(by_date)} to {max(by_date)}")

# %% [markdown]
# ## Clip and mosaic each date onto a common AOI grid
#
# Each SWIM-WE `data` asset is a COG with pixel values `0` = no water, `1` = water, `254` = cloud/invalid, `255` = nodata. We read only the AOI window from each remote COG (via `/vsicurl` range requests — no full-file downloads) reprojected onto one fixed grid, so multiple tiles on the same day mosaic together pixel-for-pixel. This step makes network requests and can take a few minutes; the result is cached to `CACHE_FILE` so re-running the notebook is instant.

# %%
west, south, east, north = AOI_BBOX
grid_width = round((east - west) / RESOLUTION_DEG)
grid_height = round((north - south) / RESOLUTION_DEG)
grid_transform = Affine(RESOLUTION_DEG, 0, west, 0, -RESOLUTION_DEG, north)
print(f"Output grid: {grid_width} x {grid_height} pixels")


def read_aoi(href: str) -> np.ndarray:
    with rasterio.open(href) as src, WarpedVRT(
        src,
        crs="EPSG:4326",
        transform=grid_transform,
        width=grid_width,
        height=grid_height,
        resampling=Resampling.nearest,
    ) as vrt:
        return vrt.read(1)


if CACHE_FILE.exists():
    cached = np.load(CACHE_FILE, allow_pickle=False)
    cube, frame_dates = cached["cube"], list(cached["dates"])
    print(f"Loaded cached cube {cube.shape} from {CACHE_FILE}")
else:
    frames, frame_dates = [], []
    t0 = time.time()
    for i, date in enumerate(sorted(by_date)):
        merged = np.full((grid_height, grid_width), 255, dtype=np.uint8)
        for item in by_date[date]:
            tile = read_aoi(item.assets["data"].href)
            fill = (merged == 255) & (tile != 255)
            merged[fill] = tile[fill]
        coverage = np.mean(merged != 255)
        if coverage >= MIN_AOI_COVERAGE:
            frames.append(merged)
            frame_dates.append(date.isoformat())
        if (i + 1) % 10 == 0 or i + 1 == len(by_date):
            print(f"  processed {i + 1}/{len(by_date)} dates, kept {len(frames)}, {time.time() - t0:.0f}s elapsed")

    cube = np.stack(frames)
    np.savez_compressed(CACHE_FILE, cube=cube, dates=np.array(frame_dates))
    print(f"Built cube {cube.shape} for {len(frame_dates)} usable dates, cached to {CACHE_FILE}")

# %% [markdown]
# ## Sanity check: first and last frame

# %%
class_colors = ["#e8e4d8", "#1f77b4", "#b0b0b0", "#ffffff"]  # land, water, cloud/invalid, nodata
class_lut = np.zeros(256, dtype=np.uint8)
class_lut[0], class_lut[1], class_lut[254], class_lut[255] = 0, 1, 2, 3
cmap = ListedColormap(class_colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, idx in zip(axes, [0, -1]):
    ax.imshow(class_lut[cube[idx]], cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(f"{AOI_NAME} — {frame_dates[idx]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Animate the full time series

# %%
remapped = class_lut[cube]
gif_path = Path(f"swim_animation_{AOI_NAME.split(',')[0].replace(' ', '_').lower()}.gif")

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(remapped[0], cmap=cmap, norm=norm, interpolation="nearest")
title = ax.set_title(f"{AOI_NAME} — {frame_dates[0]}")
ax.axis("off")
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in class_colors]
ax.legend(legend_handles, ["land", "water", "cloud/invalid", "no data"],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=8)


def update(i):
    im.set_data(remapped[i])
    title.set_text(f"{AOI_NAME} — {frame_dates[i]}")
    return im, title


anim = FuncAnimation(fig, update, frames=len(remapped), interval=400, blit=False)
anim.save(gif_path, writer=PillowWriter(fps=2))
plt.close(fig)
print(f"Saved {gif_path} ({len(remapped)} frames)")
IPyImage(filename=str(gif_path))

# %% [markdown]
# ## Bonus: water extent trend
#
# A simple time series of the water fraction within the AOI (over pixels that were actually classified, i.e. excluding cloud/nodata) — useful for spotting the lake-level trend that the animation shows visually.

# %%
import pandas as pd

water = (cube == 1).sum(axis=(1, 2))
land = (cube == 0).sum(axis=(1, 2))
water_frac = water / (water + land)

series = pd.Series(water_frac, index=pd.to_datetime(frame_dates), name="water_fraction")
ax = series.plot(marker="o", figsize=(9, 4), title=f"{AOI_NAME} — water fraction of classified AOI pixels")
ax.set_ylabel("water fraction")
ax.set_xlabel("date")
plt.tight_layout()
plt.show()

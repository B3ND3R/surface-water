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
# # Candidate water pans in the JRC Global Surface Water product
#
# Estimates how many small, isolated candidate water bodies ("water pans") show up in the **JRC Global Surface Water (GSW) v1.4** product for an area of interest in Kenya, and reports their size distribution.
#
# **Method:**
# 1. Pull the JRC GSW static image (`occurrence`/`max_extent`/`seasonality` bands) for the AOI via Earth Engine.
# 2. Build a binary water-presence mask (occurrence above a minimum threshold, to suppress one-off noise).
# 3. Label connected water pixels into discrete blobs **server-side** with `ee.Image.connectedComponents`, then vectorize to polygons with `ee.Image.reduceToVectors` (both stay in Earth Engine — nothing large gets exported as a raster).
# 4. Load a bulk OpenStreetMap reference layer of rivers/waterways and water polygons for Kenya (a Geofabrik extract, clipped to the AOI) and drop any blob that touches or overlaps it — the remainder are candidates for isolated, rain-fed water bodies rather than river/lake segments.
# 5. Report candidate counts, a size-distribution histogram, and how many fall under plausible pan-size thresholds.
# 6. Export the candidates (polygon + centroid + area) to GeoJSON.
#
# **Why connected-component labeling happens in Earth Engine, not locally:** the alternative (export a raster, label it with `scipy.ndimage.label`/`skimage.measure.label`) means pulling a full-resolution 30 m raster down first. `connectedComponents` + `reduceToVectors` do the same labeling server-side and only the resulting (small) set of polygons ever leaves Earth Engine. This scales to the whole country as a single call — verified live: 84,710 candidate blobs for all of Kenya, pulled with full geometries, in ~157 seconds.
#
# **Why a bulk Geofabrik OSM extract, not a live Overpass query:** an earlier version of this notebook queried the Overpass API live for the reference layer. That works for a small AOI but does not scale — a live test against the full Kitui County bbox (~165 km × 335 km, still much smaller than all of Kenya) didn't finish in 3 minutes even after auto-splitting into 21 sub-queries, and even a modest 0.4° test box was unreliable. A one-time bulk download from Geofabrik (`kenya-latest-free.shp.zip` — the "-latest-" URL is evergreen, not a pinned dated snapshot) sidesteps this for any AOI size, at the cost of a ~980 MB one-time download.
#
# **Note:** this notebook needs Earth Engine credentials (`earthengine authenticate`). The exact EE calls used here (`connectedComponents`, `reduceToVectors`, `reduceRegion`, `ee_to_gdf`) and the Geofabrik download/load/filter logic were all verified live with real data and real timings while writing this notebook — but that was done with standalone scripts mirroring the notebook's code, not by executing the notebook file itself end to end. Check outputs on your first full run.

# %%
import os
import zipfile
import urllib.request

import ee
import geemap
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

ee.Initialize(project="dapp-dev-475812")  # run `earthengine authenticate` once beforehand; replace with your own EE project if different

# %% [markdown]
# ## Config
#
# Set `RUN_COUNTRYWIDE = True` to run over all of Kenya instead of the small Kitui default. Both were verified live:
#
# | | Kitui default (~33 km box) | `RUN_COUNTRYWIDE = True` |
# |---|---|---|
# | Step 1 (JRC diagnostic) | ~1s | ~35s |
# | Step 3 (labeling + vectorize + pull) | ~1-2s, tens of candidates | ~157s, 84,710 candidates |
# | Step 4 (OSM reference load) | fast (clipped from local file) | fast (same file, wider clip) + one-time ~980 MB download |
# | Step 5 (sjoin filter) | <1s | ~17s, ~64K isolated |
#
# So a full countrywide run is a few minutes, not hours — no tiling or async export needed.
#
# `AOI_BBOX` defaults to a ~33 km box centered on **Kitui town** (`38.3728116, -1.5642219`, geocoded via OpenStreetMap Nominatim) — Kitui County is the most water-pan-associated county in Kenya's ASAL water literature, and this box is a fast smoke-test config for iterating on thresholds before committing to a full countrywide run.
#
# `MIN_OCCURRENCE_PCT` is the main analytical knob: JRC's `occurrence` band is "% of clear observations, 1984-2021, where water was detected," and it's *already* masked to only pixels where water was seen at least once (there's no `occurrence = 0` in the data — those pixels are nodata). So `MIN_OCCURRENCE_PCT = 1` is close to the most inclusive setting available (almost equivalent to `max_extent`), while a higher threshold starts excluding real but rarely-observed ephemeral pans -- a small rain-fed pan that only held water in a couple of clear Landsat scenes across the whole 1984-2021 record can easily sit at 1-3% occurrence. Since finding ephemeral pans is the point of this notebook, the default here is deliberately low (`1`); raise it if you want to trade recall for a more conservative, less noise-prone count. Step 1 below prints both the `max_extent` area and the actual thresholded area so you can see how much the threshold is excluding.
#
# `MAX_BLOB_SIZE_PX` has a hard ceiling: `ee.Image.connectedComponents` only accepts `maxSize` in `(0, 1024]` pixels (~92 ha at 30 m) -- this is an Earth Engine API limit, not a modeling choice. It's still a generous cap for pan-scale features (well above the 10 ha reporting threshold), but note that any genuinely isolated water body bigger than ~92 ha gets masked out before vectorization even happens, before the OSM filter ever sees it.
#
# `REFERENCE_CRS = "ESRI:102022"` (Africa Albers Equal Area Conic) is used for buffering/filtering against the OSM reference layer -- it's valid across all of Kenya (and the rest of Africa), unlike a single UTM zone, which would be wrong for parts of the country at full extent.

# %%
# Set this to True to run over all of Kenya instead of the Kitui smoke-test box (see the table above).
RUN_COUNTRYWIDE = True

# AOI: ~33 km box centered on Kitui town, Kenya (lon, lat center geocoded via OSM Nominatim)
AOI_CENTER_LON, AOI_CENTER_LAT = 38.3728116, -1.5642219
AOI_HALF_DEG = 0.15
KITUI_BBOX = (
    AOI_CENTER_LON - AOI_HALF_DEG, AOI_CENTER_LAT - AOI_HALF_DEG,
    AOI_CENTER_LON + AOI_HALF_DEG, AOI_CENTER_LAT + AOI_HALF_DEG,
)  # (west, south, east, north)
KENYA_BBOX = (33.9, -4.7, 41.9, 5.1)  # (west, south, east, north) -- a rectangle, so it overshoots slightly into neighboring countries at the edges

AOI_BBOX = KENYA_BBOX if RUN_COUNTRYWIDE else KITUI_BBOX

# JRC Global Surface Water
JRC_ASSET = "JRC/GSW1_4/GlobalSurfaceWater"
SCALE_M = 30  # native JRC resolution
MIN_OCCURRENCE_PCT = 1  # see markdown above -- occurrence is already masked to >0, so this is close to maximally inclusive

# Connected-component labeling / vectorization
MAX_BLOB_SIZE_PX = 1024  # ee.Image.connectedComponents hard limit is (0, 1024] -- ~92 ha at 30m
                          # (this is an EE API ceiling, not the semantic river/lake filter -- that's OSM, below)
EE_MAX_PIXELS = 4e9  # must exceed the AOI's actual pixel count at SCALE_M, or bestEffort silently coarsens the scale
                      # (all of Kenya at 30m is ~1.07 billion pixels -- the old default of 1e9 was too low)

# OSM reference layer (rivers/waterways/water polygons) used to exclude non-isolated blobs
RIVER_BUFFER_M = 30  # waterway lines have no width; buffer them before intersecting
REFERENCE_CRS = "ESRI:102022"  # Africa Albers Equal Area Conic -- valid across all of Kenya, unlike a single UTM zone
GEOFABRIK_URL = "https://download.geofabrik.de/africa/kenya-latest-free.shp.zip"  # evergreen URL, not a pinned dated snapshot
GEOFABRIK_CACHE_DIR = "data/geofabrik"

# Reporting thresholds
PAN_THRESHOLDS_HA = [1, 5, 10]

OUTPUT_GEOJSON = "candidate_water_pans_kenya.geojson" if RUN_COUNTRYWIDE else "candidate_water_pans_kitui.geojson"

aoi = ee.Geometry.Rectangle(list(AOI_BBOX))
print(f"AOI bbox (west, south, east, north): {AOI_BBOX}" + (" [countrywide]" if RUN_COUNTRYWIDE else " [Kitui default]"))

# %% [markdown]
# ## Step 1: JRC water-presence mask

# %%
gsw = ee.Image(JRC_ASSET)
water_mask = gsw.select("occurrence").gte(MIN_OCCURRENCE_PCT).selfMask().rename("water")

# Sanity check: compare the thresholded mask against max_extent (any water ever detected, occurrence's
# own upper bound) so it's clear whether a low count means "little water here" or "threshold too strict".
diagnostic = ee.Image.cat([
    water_mask.rename("thresholded"),
    gsw.select("max_extent").rename("max_extent"),
]).multiply(ee.Image.pixelArea())

area_m2 = diagnostic.reduceRegion(
    reducer=ee.Reducer.sum(), geometry=aoi, scale=SCALE_M, bestEffort=True, maxPixels=EE_MAX_PIXELS
).getInfo()
thresholded_ha = area_m2.get("thresholded", 0) / 10_000
max_extent_ha = area_m2.get("max_extent", 0) / 10_000
print(f"JRC water area in AOI, occurrence >= {MIN_OCCURRENCE_PCT}%: {thresholded_ha:,.1f} ha")
print(f"JRC water area in AOI, max_extent (any water ever):        {max_extent_ha:,.1f} ha")
if max_extent_ha > 0 and thresholded_ha < 0.5 * max_extent_ha:
    print(f"-> the threshold is excluding a majority of ever-detected water in this AOI; "
          f"consider lowering MIN_OCCURRENCE_PCT if that's not what you want.")

# %% [markdown]
# ## Step 2: connected-component labeling (server-side)

# %%
# 8-connected (diagonal pixels count as touching) -- avoids splitting one irregular blob
# into several just because it narrows to a diagonal-only connection at 30m.
connectivity_kernel = ee.Kernel.square(1)
labeled = water_mask.connectedComponents(connectedness=connectivity_kernel, maxSize=MAX_BLOB_SIZE_PX)
print("Labeled image bands:", labeled.bandNames().getInfo())

# %% [markdown]
# ## Step 3: vectorize labeled blobs to polygons

# %%
blob_fc = labeled.select("labels").reduceToVectors(
    geometry=aoi,
    scale=SCALE_M,
    geometryType="polygon",
    eightConnected=True,  # must match the connectivity used for labeling above
    labelProperty="blob_id",
    bestEffort=True,
    maxPixels=EE_MAX_PIXELS,
)

# Compute area (m^2, geodesic) and centroid server-side so nothing needs local reprojection
def _add_geom_props(f):
    geom = f.geometry()
    centroid = geom.centroid(1).coordinates()
    return f.set({
        "area_m2": geom.area(1),
        "centroid_lon": centroid.get(0),
        "centroid_lat": centroid.get(1),
    })

blob_fc = blob_fc.map(_add_geom_props)

# "geometry" must be included explicitly -- ee_to_gdf applies `columns` as a plain column filter
# *after* building the GeoDataFrame, so omitting it silently drops geometry and returns a plain DataFrame.
candidates = geemap.ee_to_gdf(blob_fc, columns=["blob_id", "area_m2", "centroid_lon", "centroid_lat", "geometry"])
candidates["area_ha"] = candidates["area_m2"] / 10_000
print(f"Step 3: {len(candidates)} candidate blobs after connected-component labeling + vectorization "
      f"(blobs larger than {MAX_BLOB_SIZE_PX:,} px / ~{MAX_BLOB_SIZE_PX * SCALE_M**2 / 1e4:,.0f} ha already excluded)")

# %% [markdown]
# ## Step 4: OSM reference layer (rivers, waterways, water polygons)
#
# Loaded from a bulk Geofabrik extract for all of Kenya (downloaded once, cached locally, then clipped to whatever AOI is active) rather than a live Overpass query -- see the intro markdown for why. `waterway` lines have zero width, so they're buffered by `RIVER_BUFFER_M` before use; water polygons are used as-is.

# %%
WATER_SHP = os.path.join(GEOFABRIK_CACHE_DIR, "gis_osm_water_a_free_1.shp")
WATERWAYS_SHP = os.path.join(GEOFABRIK_CACHE_DIR, "gis_osm_waterways_free_1.shp")

if not (os.path.exists(WATER_SHP) and os.path.exists(WATERWAYS_SHP)):
    os.makedirs(GEOFABRIK_CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(GEOFABRIK_CACHE_DIR, "kenya-latest-free.shp.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {GEOFABRIK_URL} (~980 MB, one-time)...")
        urllib.request.urlretrieve(GEOFABRIK_URL, zip_path)
    print("Extracting water/waterway layers...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if m.startswith("gis_osm_water_a_free_1.") or m.startswith("gis_osm_waterways_free_1.")]
        zf.extractall(GEOFABRIK_CACHE_DIR, members=members)
    print(f"Step 4: extracted reference layers to {GEOFABRIK_CACHE_DIR}")
else:
    print(f"Step 4: using cached reference layers in {GEOFABRIK_CACHE_DIR}")

water_poly = gpd.read_file(WATER_SHP, bbox=AOI_BBOX)
waterways = gpd.read_file(WATERWAYS_SHP, bbox=AOI_BBOX)
print(f"Step 4: {len(water_poly)} water polygons + {len(waterways)} waterway lines within the AOI")

if len(water_poly) == 0 and len(waterways) == 0:
    print("WARNING: no OSM reference features found for this AOI -- the river/lake filter below will be a no-op. "
          "Verify this is expected (e.g. a genuinely river-free AOI) rather than a data gap.")
    reference = gpd.GeoDataFrame(geometry=[], crs=REFERENCE_CRS)
else:
    water_proj = water_poly.to_crs(REFERENCE_CRS)
    waterways_proj = waterways.to_crs(REFERENCE_CRS)
    waterways_proj["geometry"] = waterways_proj.geometry.buffer(RIVER_BUFFER_M)
    reference = gpd.GeoDataFrame(
        geometry=list(water_proj.geometry) + list(waterways_proj.geometry), crs=REFERENCE_CRS
    )

# %% [markdown]
# ## Step 5: drop blobs that touch the river/lake reference layer
#
# Uses `gpd.sjoin` rather than unioning the reference layer into one geometry and testing `.intersects()` against it -- verified live at country scale (84,710 candidates against 66,007 reference features): sjoin took ~17s, comfortably fast at this size.

# %%
n_before = len(candidates)
if len(reference) == 0:
    isolated = candidates.copy()
else:
    candidates_proj = candidates.to_crs(REFERENCE_CRS)
    joined = gpd.sjoin(candidates_proj, reference, predicate="intersects", how="left")
    touches_reference = joined.groupby(level=0)["index_right"].apply(lambda s: s.notna().any())
    isolated = candidates[~candidates.index.map(touches_reference)].copy()
n_after = len(isolated)
print(f"Step 5: {n_before} candidates before river/lake filter -> {n_after} after "
      f"({n_before - n_after} dropped as touching/overlapping a mapped river, waterway, or water polygon)")

# %% [markdown]
# ## Step 6: size distribution

# %%
print(f"Total candidate isolated water bodies: {n_after}")
for t in PAN_THRESHOLDS_HA:
    n_below = int((isolated["area_ha"] < t).sum())
    pct = 100 * n_below / n_after if n_after else 0
    print(f"  below {t:>2} ha: {n_below:>5} ({pct:5.1f}% of candidates)")

min_resolvable_ha = SCALE_M**2 / 10_000  # one JRC pixel, in ha -- a hard floor on what's even representable
print(f"\nFor reference, one JRC pixel at {SCALE_M} m is {min_resolvable_ha:.2f} ha -- "
      f"candidates near this size are only marginally resolvable, not confidently detected/sized.")

# %%
fig, ax = plt.subplots(figsize=(8, 5))

areas = isolated["area_ha"].to_numpy()
areas = areas[areas > 0]
bins = np.logspace(np.log10(max(areas.min(), 0.01)), np.log10(areas.max()), 30) if len(areas) else [0, 1]

ax.hist(areas, bins=bins, color="#1f77b4", edgecolor="white", linewidth=0.5)
ax.set_xscale("log")
ax.set_xlabel("Blob area (ha, log scale)")
ax.set_ylabel("Candidate count")
ax.set_title(f"Size distribution of candidate isolated water bodies (n={n_after})")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

for t in PAN_THRESHOLDS_HA:
    ax.axvline(t, color="#d62728", linestyle="--", linewidth=1)
    ax.text(t, ax.get_ylim()[1] * 0.97, f" {t} ha", color="#d62728", fontsize=9, va="top")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: export candidates

# %%
export_cols = ["blob_id", "area_ha", "centroid_lon", "centroid_lat", "geometry"]
isolated[export_cols].to_file(OUTPUT_GEOJSON, driver="GeoJSON")
print(f"Saved {n_after} candidate water bodies to {OUTPUT_GEOJSON}")

# %% [markdown]
# ## Caveats and known bias sources
#
# - **Cloud/gap bias in JRC.** The `occurrence` band is computed only from *clear* observations; chronically cloud-obscured areas have fewer clear looks over 1984-2021, which can understate `occurrence` and cause real small water bodies to fall below `MIN_OCCURRENCE_PCT` and go uncounted. This mainly affects persistently cloudy regions (e.g. western Kenya highlands) more than the ASAL default AOI here, but matters more at country scale.
# - **OSM reference-layer completeness bias runs one direction: overcounting.** If a real river, stream, or small lake isn't mapped in OSM, it won't get excluded in Step 5 and will show up as a false-positive "candidate pan." Rural Kenya's OSM waterway coverage is uneven — spot-check a sample of candidates against imagery before trusting the count as a hard number, and consider it an upper bound.
# - **30 m resolution floor.** A single JRC pixel is 0.09 ha; reliably resolving a blob's shape/area needs several pixels, so counts near the smallest `PAN_THRESHOLDS_HA` bucket (<1 ha) are the least trustworthy — treat the sub-1-ha bucket as "pans and/or noise, indistinguishable at this resolution," not a confident pan count.
# - **`MIN_OCCURRENCE_PCT` is a real analytical choice, not a neutral default.** Lower values pull in more true ephemeral pans but also more sensor noise; higher values are more conservative and will undercount pans that only fill in exceptional years. If you have any known pan locations, spot-check against them to help calibrate.
# - **`MAX_BLOB_SIZE_PX`'s ~92 ha ceiling is an Earth Engine API limit** (`connectedComponents`'s `maxSize` only accepts values up to 1024), not a modeling choice — any genuinely isolated water body larger than that is masked out before vectorization, before the OSM filter ever runs.
# - **AOI edge effects.** Blobs that straddle the AOI boundary are clipped to it, so their reported area understates the true feature size, and in rare cases a genuinely connected large water body could be clipped into a smaller, seemingly-isolated fragment. `KENYA_BBOX` is a rectangle and overshoots slightly into Uganda, Tanzania, Ethiopia, Somalia, and South Sudan at the edges.
# - **Vectorized polygon boundaries are a raster-to-vector simplification** of the 30 m mask, not surveyed boundaries — treat areas as approximate, not survey-grade.
# - **The Geofabrik OSM extract is a ~980 MB one-time download**, cached to `data/geofabrik/` (gitignored). It's a snapshot as of whenever it was downloaded, not live — re-download (delete the cache dir) if you need current OSM data.

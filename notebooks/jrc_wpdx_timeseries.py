# %%
"""
JRC Global Surface Water + WPDx overlay time series
=====================================================

What this does:
  1. Pulls Kenya water-point locations from the Water Point Data Exchange (WPDx).
  2. Pulls the JRC Global Surface Water Monthly History image collection for the
     same area.
  3. Builds an interactive map with a time slider that "plays" the monthly JRC
     water layer, with the WPDx asset points overlaid on top.
  4. Also samples the JRC classification at each WPDx point for every month, so
     you get a per-asset time series table (not just a visual), and optionally
     exports the animation as a standalone GIF (useful outside Jupyter).

Best run in Jupyter / Google Colab, since the interactive time slider
(geemap.Map.add_time_slider) needs a notebook front end. The GIF export at the
bottom works from a plain terminal too.

Prerequisites (not included/verifiable in this sandbox):
  pip install earthengine-api geemap pandas requests --break-system-packages
  earthengine authenticate   # one-time, needs a Google/Earth Engine account

WPDx access notes:
  WPDx data is served through a Socrata (SODA) API at data.waterpointdata.org.
  The dataset resource id below (WPDX_RESOURCE_ID) is the WPdx+ dataset as
  referenced in Socrata's developer docs. Socrata field names occasionally
  change, so this script prints the columns it actually receives -- check
  those against WATER_LAT_FIELD / WATER_LON_FIELD / WATER_SOURCE_FIELD below
  and adjust if your pull looks empty or mis-mapped.
"""

# %%
import ee
import geemap
import pandas as pd
import requests
from IPython.display import display

# %% [markdown]
# ---------------------------------------------------------------------------
# 0. CONFIG -- adjust these for your area/date range
# ---------------------------------------------------------------------------

# %%
COUNTRY_NAME = "Kenya"
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

# %%
# Bounding box fallback if you don't want to wait on the WPDx pull first.
# Rough Kenya bounding box: [min_lon, min_lat, max_lon, max_lat]
KENYA_BBOX = [33.9, -4.7, 41.9, 5.1]

# %%
WPDX_RESOURCE_ID = "eqje-vguj"          # WPdx+ Socrata dataset id -- verify at
                                         # https://dev.socrata.com/foundry/data.waterpointdata.org/eqje-vguj
WPDX_BASE_URL = f"https://data.waterpointdata.org/resource/{WPDX_RESOURCE_ID}.json"

# %%
# Common WPDx Data Standard field names -- confirm against the printed
# column list in step 1 and adjust if your pull differs.
COUNTRY_FIELD = "clean_country_name"
LAT_FIELD = "lat_deg"
LON_FIELD = "lon_deg"
SOURCE_FIELD = "water_source_clean"
STATUS_FIELD = "status_id"
ID_FIELD = "wpdx_id"

# %%
MAX_POINTS = 2000   # cap the pull so the demo stays fast; raise as needed

# %% [markdown]
# ---------------------------------------------------------------------------
# 1. PULL WPDX ASSET DATA
# ---------------------------------------------------------------------------

# %%
def fetch_wpdx(country=COUNTRY_NAME, limit=MAX_POINTS):
    params = {
        "$where": f"{COUNTRY_FIELD}='{country}'",
        "$limit": limit,
    }
    resp = requests.get(WPDX_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    records = resp.json()
    df = pd.DataFrame.from_records(records)

    print(f"Pulled {len(df)} WPDx records for {country}.")
    print("Columns returned:", list(df.columns))

    for field in (LAT_FIELD, LON_FIELD):
        if field not in df.columns:
            raise KeyError(
                f"Expected field '{field}' not found. Check the printed "
                f"column list above and update the *_FIELD constants."
            )

    df[LAT_FIELD] = pd.to_numeric(df[LAT_FIELD], errors="coerce")
    df[LON_FIELD] = pd.to_numeric(df[LON_FIELD], errors="coerce")
    df = df.dropna(subset=[LAT_FIELD, LON_FIELD])
    return df


# %%
def _clean_prop(value):
    """Coerce missing/NaN values to '' -- raw NaN isn't valid JSON and makes
    Earth Engine reject the whole FeatureCollection payload when rendering."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value


def wpdx_to_ee(df):
    """Convert the WPDx dataframe into an ee.FeatureCollection of points."""
    features = []
    for _, row in df.iterrows():
        geom = ee.Geometry.Point([row[LON_FIELD], row[LAT_FIELD]])
        props = {
            "id": _clean_prop(row.get(ID_FIELD, "")),
            "source": _clean_prop(row.get(SOURCE_FIELD, "")),
            "status": _clean_prop(row.get(STATUS_FIELD, "")),
        }
        features.append(ee.Feature(geom, props))
    return ee.FeatureCollection(features)


# %% [markdown]
# ---------------------------------------------------------------------------
# 2. PULL JRC GLOBAL SURFACE WATER MONTHLY HISTORY
# ---------------------------------------------------------------------------

# %%
def get_jrc_monthly(aoi, start_date=START_DATE, end_date=END_DATE):
    """
    JRC/GSW1_4/MonthlyHistory encodes each pixel per month as:
      0 = no observation, 1 = not water, 2 = water
    """
    collection = (
        ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
    )
    return collection


# %%
JRC_VIS = {"min": 0, "max": 2, "palette": ["black", "sienna", "blue"]}


# %% [markdown]
# ---------------------------------------------------------------------------
# 3. INTERACTIVE MAP: JRC TIME SLIDER + WPDX OVERLAY
# ---------------------------------------------------------------------------

# %%
def build_interactive_map(jrc_collection, wpdx_fc, aoi):
    m = geemap.Map()
    m.centerObject(aoi, zoom=6)

    # Animated JRC layer with a built-in time slider (Jupyter/Colab only)
    m.add_time_slider(jrc_collection, JRC_VIS, layer_name="JRC Monthly Water", time_interval=1)

    # WPDx points stay fixed on top of the slider layer
    m.addLayer(wpdx_fc, {"color": "red"}, "WPDx assets")

    return m


# %% [markdown]
# ---------------------------------------------------------------------------
# 4. PER-ASSET TIME SERIES TABLE (works outside Jupyter too)
# ---------------------------------------------------------------------------

# %%
def sample_jrc_at_points(jrc_collection, wpdx_fc, scale=30):
    """
    For each month in the JRC collection, sample the water classification at
    every WPDx point. Returns a long-format DataFrame:
      wpdx_id | date | jrc_value  (0=no obs, 1=not water, 2=water)
    """
    image_list = jrc_collection.toList(jrc_collection.size())
    n_images = jrc_collection.size().getInfo()

    rows = []
    for i in range(n_images):
        img = ee.Image(image_list.get(i))
        date = img.date().format("YYYY-MM").getInfo()
        sampled = img.sampleRegions(collection=wpdx_fc, scale=scale, geometries=False)
        for feat in sampled.getInfo()["features"]:
            props = feat["properties"]
            rows.append({
                "wpdx_id": props.get("id"),
                "date": date,
                "jrc_value": props.get("water"),  # band name from GSW1_4
            })
        if i % 12 == 0:
            print(f"  sampled {i}/{n_images} months...")

    return pd.DataFrame(rows)


# %% [markdown]
# ---------------------------------------------------------------------------
# 5. STANDALONE GIF EXPORT (no Jupyter required)
# ---------------------------------------------------------------------------

# %%
def export_gif(jrc_collection, aoi, out_path="jrc_timeseries.gif", frames_per_second=3):
    """
    Renders the JRC monthly collection as a GIF clipped to aoi. WPDx points are
    not burned into this raster GIF (geemap's raster GIF export doesn't overlay
    vectors) -- use the interactive map above if you need points visible frame
    by frame, or see the note below for burning points in manually.
    """
    geemap.download_ee_video(
        jrc_collection,
        {"region": aoi, "dimensions": 768, **JRC_VIS, "framesPerSecond": frames_per_second},
        out_path,
    )
    print(f"Saved animation to {out_path}")


# %% [markdown]
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# %%
if __name__ == "__main__":
    ee.Initialize(project="dapp-dev-475812")  # run `earthengine authenticate` once beforehand

    print("Step 1: pulling WPDx data...")
    wpdx_df = fetch_wpdx()
    wpdx_fc = wpdx_to_ee(wpdx_df)

    aoi = ee.Geometry.Rectangle(KENYA_BBOX)

    print("Step 2: pulling JRC Global Surface Water Monthly History...")
    jrc = get_jrc_monthly(aoi)

    print("Step 3: building interactive map (open in Jupyter to see the time slider)...")
    m = build_interactive_map(jrc, wpdx_fc, aoi)
    display(m)  # bare `m` only auto-displays if it's the cell's last statement -- it isn't here

    print("Step 4: sampling JRC values at each WPDx point (this can take a while)...")
    # Tip: restrict jrc/wpdx_fc to a small county or a handful of named assets
    # before running this at scale -- sampling every month at every point
    # nationally will be slow.
    # timeseries_df = sample_jrc_at_points(jrc, wpdx_fc)
    # timeseries_df.to_csv("jrc_wpdx_timeseries.csv", index=False)

    print("Step 5 (optional): export a standalone GIF of the JRC layer.")
    # export_gif(jrc, aoi)

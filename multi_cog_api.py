import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rio_tiler.colormap import cmap
from rio_tiler.io import COGReader


# ============================================================
# Config
# ============================================================

APP_TITLE = "Multi COG Raster API"
APP_VERSION = "1.1.0"

BASE_DIR = Path(__file__).resolve().parent

# Render-safe path.
# Default expects:
# data/output/{dataset_name}/*.tif
COG_ROOT = Path(os.getenv("COG_ROOT", "data/output")).resolve()

# Optional. If empty, API uses current request domain.
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")


SUPPORTED_DATASETS = {
    "rain_ssp245_1540_p95_cog": {
        "title": "Rainfall SSP2-4.5 2015-2040 P95",
        "variable": "rainfall",
        "unit": "mm",
        "default_rescale": [0, 300],
        "default_colormap": "turbo",
    },
    "rain_ssp245_4160_p95_cog": {
        "title": "Rainfall SSP2-4.5 2041-2060 P95",
        "variable": "rainfall",
        "unit": "mm",
        "default_rescale": [0, 300],
        "default_colormap": "turbo",
    },
    "rain_ssp585_1540_p95_cog": {
        "title": "Rainfall SSP5-8.5 2015-2040 P95",
        "variable": "rainfall",
        "unit": "mm",
        "default_rescale": [0, 300],
        "default_colormap": "turbo",
    },
    "rain_ssp585_4160_p95_cog": {
        "title": "Rainfall SSP5-8.5 2041-2060 P95",
        "variable": "rainfall",
        "unit": "mm",
        "default_rescale": [0, 300],
        "default_colormap": "turbo",
    },
    "temp_ssp245_1540_p95_cog": {
        "title": "Temperature SSP2-4.5 2015-2040 P95",
        "variable": "temperature",
        "unit": "°C",
        "default_rescale": [20, 45],
        "default_colormap": "turbo",
    },
    "temp_ssp245_4160_p95_cog": {
        "title": "Temperature SSP2-4.5 2041-2060 P95",
        "variable": "temperature",
        "unit": "°C",
        "default_rescale": [20, 45],
        "default_colormap": "turbo",
    },
    "temp_ssp585_1540_p95_cog": {
        "title": "Temperature SSP5-8.5 2015-2040 P95",
        "variable": "temperature",
        "unit": "°C",
        "default_rescale": [20, 45],
        "default_colormap": "turbo",
    },
    "temp_ssp585_4160_p95_cog": {
        "title": "Temperature SSP5-8.5 2041-2060 P95",
        "variable": "temperature",
        "unit": "°C",
        "default_rescale": [20, 45],
        "default_colormap": "turbo",
    },
}


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


# ============================================================
# App
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="FastAPI service for serving monthly Cloud Optimized GeoTIFF raster tiles.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ============================================================
# General helpers
# ============================================================

def get_public_base_url(request: Optional[Request] = None) -> str:
    if API_BASE_URL:
        return API_BASE_URL

    if request:
        return str(request.base_url).rstrip("/")

    return ""


def validate_dataset(dataset: str) -> None:
    if dataset not in SUPPORTED_DATASETS:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Dataset '{dataset}' is not supported.",
                "available_datasets": list(SUPPORTED_DATASETS.keys()),
            },
        )


def get_dataset_config(dataset: str) -> Dict:
    validate_dataset(dataset)
    return SUPPORTED_DATASETS[dataset]


def get_colormap(colormap_name: str):
    try:
        return cmap.get(colormap_name)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid colormap_name: {colormap_name}",
                "examples": ["turbo", "viridis", "plasma", "inferno", "magma"],
            },
        )


def safe_float(value) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


# ============================================================
# File / month helpers
# ============================================================

def list_dataset_tifs(dataset: str) -> List[Path]:
    validate_dataset(dataset)

    dataset_dir = COG_ROOT / dataset

    if not dataset_dir.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Dataset folder not found: {dataset_dir}",
                "hint": "Check COG_ROOT and your data/output folder.",
            },
        )

    tif_files = sorted(
        list(dataset_dir.glob("*.tif"))
        + list(dataset_dir.glob("*.tiff"))
        + list(dataset_dir.glob("*.TIF"))
        + list(dataset_dir.glob("*.TIFF"))
    )

    if not tif_files:
        raise HTTPException(
            status_code=404,
            detail=f"No GeoTIFF or COG files found in {dataset_dir}",
        )

    return tif_files


def find_month_cog(dataset: str, month: int) -> Path:
    """
    Find the COG file for a given month.

    Supports:
    1. One file per month:
       something_month_01_cog.tif
       something_m01_cog.tif
       something_01_cog.tif

    2. One 12-band file in folder:
       if only one COG exists, this returns that file and band index is resolved later.
    """
    validate_dataset(dataset)

    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month must be between 1 and 12.")

    tif_files = list_dataset_tifs(dataset)

    # If there is only one file, it may be a 12-band COG.
    if len(tif_files) == 1:
        return tif_files[0]

    month_patterns = [
        f"month_{month}",
        f"month_{month:02d}",
        f"month{month}",
        f"month{month:02d}",
        f"m_{month}",
        f"m_{month:02d}",
        f"m{month}",
        f"m{month:02d}",
        f"_{month}_",
        f"_{month:02d}_",
        f"-{month}-",
        f"-{month:02d}-",
        f"_{month}_cog",
        f"_{month:02d}_cog",
        f"-{month}-cog",
        f"-{month:02d}-cog",
    ]

    for tif in tif_files:
        stem = tif.stem.lower()
        if any(pattern in stem for pattern in month_patterns):
            return tif

    # Fallback: if exactly 12 files, use sorted order.
    if len(tif_files) == 12:
        return tif_files[month - 1]

    raise HTTPException(
        status_code=404,
        detail={
            "message": f"Cannot find month {month} COG for dataset '{dataset}'.",
            "hint": "Use filenames containing month number, or keep exactly 12 sorted monthly COG files in each dataset folder.",
            "available_files": [f.name for f in tif_files],
        },
    )


def resolve_band_index(cog_path: Path, month: int, requested_bidx: Optional[int] = None) -> int:
    """
    If the COG has 12 bands, use month as band index.
    If each month is a separate COG file, use band 1.

    User can override with bidx.
    """
    if requested_bidx is not None:
        return requested_bidx

    try:
        with rasterio.open(cog_path) as src:
            if src.count >= 12:
                return month
            return 1
    except Exception:
        return 1


def get_available_months(dataset: str) -> List[Dict]:
    months = []

    for month in range(1, 13):
        try:
            path = find_month_cog(dataset, month)
            bidx = resolve_band_index(path, month)

            months.append(
                {
                    "month": month,
                    "name": MONTH_NAMES[month],
                    "available": True,
                    "filename": path.name,
                    "band_index": bidx,
                }
            )
        except HTTPException:
            months.append(
                {
                    "month": month,
                    "name": MONTH_NAMES[month],
                    "available": False,
                    "filename": None,
                    "band_index": None,
                }
            )

    return months


# ============================================================
# Statistics / rescale helpers
# ============================================================

@lru_cache(maxsize=512)
def compute_cog_stats_cached(cog_path_str: str, bidx: int = 1) -> Dict:
    """
    Calculate approximate statistics from the actual deployed COG.

    This is important because QGIS auto-stretches rasters,
    but web tile rendering needs explicit rescale values.

    The function is cached so the same dataset/month is not recalculated every tile request.
    """
    cog_path = Path(cog_path_str)

    try:
        with rasterio.open(cog_path) as src:
            if bidx < 1 or bidx > src.count:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid bidx={bidx}. File has {src.count} band(s).",
                )

            max_size = 1024
            scale = max(src.width / max_size, src.height / max_size, 1)

            out_width = max(1, int(src.width / scale))
            out_height = max(1, int(src.height / scale))

            arr = src.read(
                bidx,
                out_shape=(out_height, out_width),
                masked=True,
                resampling=Resampling.bilinear,
            )

            data = arr.compressed()
            data = data[np.isfinite(data)]

            if data.size == 0:
                raise HTTPException(
                    status_code=500,
                    detail="No valid data found in raster.",
                )

            data_min = float(np.nanmin(data))
            data_max = float(np.nanmax(data))
            data_mean = float(np.nanmean(data))
            data_std = float(np.nanstd(data))

            p2 = float(np.nanpercentile(data, 2))
            p5 = float(np.nanpercentile(data, 5))
            p50 = float(np.nanpercentile(data, 50))
            p95 = float(np.nanpercentile(data, 95))
            p98 = float(np.nanpercentile(data, 98))

            return {
                "min": data_min,
                "max": data_max,
                "mean": data_mean,
                "std": data_std,
                "p2": p2,
                "p5": p5,
                "p50": p50,
                "p95": p95,
                "p98": p98,
            }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot calculate COG statistics for {cog_path.name}: {str(exc)}",
        )


def get_recommended_rescale(dataset: str, cog_path: Path, bidx: int = 1) -> Tuple[float, float]:
    """
    Dynamic rescale logic.

    Temperature:
      Use p2-p98 because temperature has a narrow range and needs contrast.

    Rainfall:
      Use 0-p98 because rainfall should remain anchored to zero.
    """
    config = get_dataset_config(dataset)
    variable = config["variable"]

    stats = compute_cog_stats_cached(str(cog_path), bidx)

    if variable == "temperature":
        rescale_min = stats["p2"]
        rescale_max = stats["p98"]

    elif variable == "rainfall":
        rescale_min = 0.0
        rescale_max = max(stats["p98"], 1.0)

    else:
        rescale_min = stats["p2"]
        rescale_max = stats["p98"]

    if rescale_min == rescale_max:
        rescale_max = rescale_min + 1.0

    return float(rescale_min), float(rescale_max)


def parse_rescale(
    rescale: Optional[str],
    dataset: str,
    cog_path: Optional[Path] = None,
    bidx: int = 1,
) -> Tuple[float, float]:
    """
    Supported:
    - rescale=auto
    - rescale=28,40
    - rescale omitted

    If omitted or auto, it uses real COG stats.
    """
    if rescale and rescale.lower() != "auto":
        try:
            parts = [float(x.strip()) for x in rescale.split(",")]

            if len(parts) != 2:
                raise ValueError

            if parts[0] == parts[1]:
                raise ValueError

            return parts[0], parts[1]

        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid rescale value. Use format: min,max or rescale=auto.",
            )

    if cog_path is not None:
        return get_recommended_rescale(dataset, cog_path, bidx=bidx)

    default_min, default_max = SUPPORTED_DATASETS[dataset]["default_rescale"]
    return float(default_min), float(default_max)


def format_rescale(rescale_min: float, rescale_max: float) -> str:
    return f"{rescale_min:.4f},{rescale_max:.4f}"


# ============================================================
# Bounds helpers
# ============================================================

def get_cog_geographic_bounds(cog) -> Tuple[float, float, float, float]:
    """
    Return bounds in EPSG:4326 for TileJSON.
    Compatible with different rio-tiler versions.
    """
    bounds = cog.bounds

    try:
        dataset_crs = cog.dataset.crs
    except Exception:
        dataset_crs = None

    if dataset_crs is None:
        return tuple(bounds)

    try:
        if dataset_crs.to_epsg() == 4326:
            return tuple(bounds)

        return transform_bounds(
            dataset_crs,
            "EPSG:4326",
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            densify_pts=21,
        )

    except Exception:
        return tuple(bounds)


# ============================================================
# URL builders
# ============================================================

def build_tile_url(
    request: Request,
    dataset: str,
    month: int,
    rescale: str,
    colormap_name: str,
    bidx: Optional[int] = None,
) -> str:
    base_url = get_public_base_url(request)

    url = (
        f"{base_url}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&rescale={rescale}"
        f"&colormap_name={colormap_name}"
    )

    if bidx is not None:
        url += f"&bidx={bidx}"

    return url


def build_example_tile_url(
    request: Request,
    dataset: str,
    month: int,
    rescale: str,
    colormap_name: str,
    bidx: Optional[int] = None,
) -> str:
    base_url = get_public_base_url(request)

    # Example tile around Thailand at zoom level 5.
    url = (
        f"{base_url}/cog/tiles/WebMercatorQuad/5/25/14.png"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&rescale={rescale}"
        f"&colormap_name={colormap_name}"
    )

    if bidx is not None:
        url += f"&bidx={bidx}"

    return url


def build_tilejson_url(
    request: Request,
    dataset: str,
    month: int,
    rescale: str,
    colormap_name: str,
    bidx: Optional[int] = None,
) -> str:
    base_url = get_public_base_url(request)

    url = (
        f"{base_url}/cog/tilejson.json"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&rescale={rescale}"
        f"&colormap_name={colormap_name}"
    )

    if bidx is not None:
        url += f"&bidx={bidx}"

    return url


# ============================================================
# UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "title": APP_TITLE,
        },
    )


# ============================================================
# Health / Metadata
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Multi COG API is running",
        "version": APP_VERSION,
        "cog_root": str(COG_ROOT),
    }


@app.get("/service")
def service_info(request: Request):
    base_url = get_public_base_url(request)

    return {
        "message": "Multi COG API is running",
        "version": APP_VERSION,
        "base_url": base_url,
        "docs": f"{base_url}/docs",
        "layers": f"{base_url}/layers",
        "datasets": f"{base_url}/datasets",
        "stats_example": f"{base_url}/cog/stats?dataset=temp_ssp585_4160_p95_cog&month=1",
        "example_tile_auto": f"{base_url}/cog/tiles/WebMercatorQuad/5/25/14.png?dataset=temp_ssp585_4160_p95_cog&month=1&rescale=auto&colormap_name=turbo",
    }


@app.get("/datasets")
def list_datasets():
    items = []

    for dataset_name, config in SUPPORTED_DATASETS.items():
        dataset_dir = COG_ROOT / dataset_name
        exists = dataset_dir.exists()

        tif_count = 0
        if exists:
            tif_count = len(
                list(dataset_dir.glob("*.tif"))
                + list(dataset_dir.glob("*.tiff"))
                + list(dataset_dir.glob("*.TIF"))
                + list(dataset_dir.glob("*.TIFF"))
            )

        items.append(
            {
                "name": dataset_name,
                "title": config["title"],
                "variable": config["variable"],
                "unit": config["unit"],
                "default_rescale": config["default_rescale"],
                "recommended_rescale_mode": "auto",
                "default_colormap": config["default_colormap"],
                "folder_exists": exists,
                "file_count": tif_count,
            }
        )

    return {
        "cog_root": str(COG_ROOT),
        "count": len(items),
        "datasets": items,
    }


@app.get("/layers")
def list_layers(request: Request):
    layers = []

    for dataset_name, config in SUPPORTED_DATASETS.items():
        months = get_available_months(dataset_name)

        for month_item in months:
            if not month_item["available"]:
                continue

            colormap_name = config["default_colormap"]
            rescale = "auto"

            layers.append(
                {
                    "id": f"{dataset_name}_month_{month_item['month']:02d}",
                    "dataset": dataset_name,
                    "title": config["title"],
                    "variable": config["variable"],
                    "unit": config["unit"],
                    "month": month_item["month"],
                    "month_name": month_item["name"],
                    "filename": month_item["filename"],
                    "band_index": month_item["band_index"],
                    "default_rescale": config["default_rescale"],
                    "recommended_rescale_mode": "auto",
                    "default_colormap": colormap_name,
                    "tile_url_template": build_tile_url(
                        request=request,
                        dataset=dataset_name,
                        month=month_item["month"],
                        rescale=rescale,
                        colormap_name=colormap_name,
                    ),
                    "example_tile_url": build_example_tile_url(
                        request=request,
                        dataset=dataset_name,
                        month=month_item["month"],
                        rescale=rescale,
                        colormap_name=colormap_name,
                    ),
                    "tilejson_url": build_tilejson_url(
                        request=request,
                        dataset=dataset_name,
                        month=month_item["month"],
                        rescale=rescale,
                        colormap_name=colormap_name,
                    ),
                    "stats_url": (
                        f"{get_public_base_url(request)}/cog/stats"
                        f"?dataset={dataset_name}"
                        f"&month={month_item['month']}"
                    ),
                }
            )

    return {
        "count": len(layers),
        "layers": layers,
    }


@app.get("/layers/{dataset}")
def dataset_layers(dataset: str, request: Request):
    validate_dataset(dataset)

    config = SUPPORTED_DATASETS[dataset]
    months = get_available_months(dataset)

    colormap_name = config["default_colormap"]
    rescale = "auto"

    return {
        "dataset": dataset,
        "title": config["title"],
        "variable": config["variable"],
        "unit": config["unit"],
        "default_rescale": config["default_rescale"],
        "recommended_rescale_mode": "auto",
        "default_colormap": config["default_colormap"],
        "months": [
            {
                **month_item,
                "tile_url_template": build_tile_url(
                    request=request,
                    dataset=dataset,
                    month=month_item["month"],
                    rescale=rescale,
                    colormap_name=colormap_name,
                )
                if month_item["available"]
                else None,
                "example_tile_url": build_example_tile_url(
                    request=request,
                    dataset=dataset,
                    month=month_item["month"],
                    rescale=rescale,
                    colormap_name=colormap_name,
                )
                if month_item["available"]
                else None,
                "tilejson_url": build_tilejson_url(
                    request=request,
                    dataset=dataset,
                    month=month_item["month"],
                    rescale=rescale,
                    colormap_name=colormap_name,
                )
                if month_item["available"]
                else None,
                "stats_url": (
                    f"{get_public_base_url(request)}/cog/stats"
                    f"?dataset={dataset}"
                    f"&month={month_item['month']}"
                )
                if month_item["available"]
                else None,
            }
            for month_item in months
        ],
    }


# ============================================================
# COG endpoints
# ============================================================

@app.get("/cog/stats")
def cog_stats(
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    stats = compute_cog_stats_cached(str(cog_path), resolved_bidx)
    rescale_min, rescale_max = get_recommended_rescale(
        dataset=dataset,
        cog_path=cog_path,
        bidx=resolved_bidx,
    )

    config = SUPPORTED_DATASETS[dataset]

    return {
        "dataset": dataset,
        "title": config["title"],
        "variable": config["variable"],
        "unit": config["unit"],
        "month": month,
        "month_name": MONTH_NAMES[month],
        "filename": cog_path.name,
        "band_index": resolved_bidx,
        "stats": stats,
        "recommended_rescale": [rescale_min, rescale_max],
        "recommended_rescale_text": format_rescale(rescale_min, rescale_max),
    }


@app.get("/cog/info")
def cog_info(
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12, description="Month number 1-12"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    try:
        with COGReader(str(cog_path)) as cog:
            info = cog.info()
            bounds = cog.bounds
            geographic_bounds = get_cog_geographic_bounds(cog)

        stats = compute_cog_stats_cached(str(cog_path), resolved_bidx)
        rescale_min, rescale_max = get_recommended_rescale(
            dataset=dataset,
            cog_path=cog_path,
            bidx=resolved_bidx,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot read COG info: {str(exc)}",
        )

    return {
        "dataset": dataset,
        "month": month,
        "month_name": MONTH_NAMES[month],
        "filename": cog_path.name,
        "path": str(cog_path),
        "band_index": resolved_bidx,
        "bounds": bounds,
        "geographic_bounds": geographic_bounds,
        "info": info,
        "stats": stats,
        "recommended_rescale": [rescale_min, rescale_max],
        "recommended_rescale_text": format_rescale(rescale_min, rescale_max),
    }


@app.get("/cog/tilejson.json")
def tilejson(
    request: Request,
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12),
    rescale: Optional[str] = Query("auto", description="Example: 0,300 or auto"),
    colormap_name: str = Query("turbo"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    rescale_min, rescale_max = parse_rescale(
        rescale=rescale,
        dataset=dataset,
        cog_path=cog_path,
        bidx=resolved_bidx,
    )
    rescale_text = format_rescale(rescale_min, rescale_max)

    tile_url = build_tile_url(
        request=request,
        dataset=dataset,
        month=month,
        rescale=rescale_text,
        colormap_name=colormap_name,
        bidx=resolved_bidx,
    )

    try:
        with COGReader(str(cog_path)) as cog:
            bounds = get_cog_geographic_bounds(cog)
            minzoom = getattr(cog, "minzoom", 0)
            maxzoom = getattr(cog, "maxzoom", 14)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cannot create TileJSON: {str(exc)}",
        )

    return {
        "tilejson": "2.2.0",
        "name": f"{dataset} - {MONTH_NAMES[month]}",
        "version": APP_VERSION,
        "scheme": "xyz",
        "tiles": [tile_url],
        "bounds": list(bounds),
        "minzoom": minzoom,
        "maxzoom": maxzoom,
        "attribution": "Multi COG Raster API",
    }


@app.get("/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png")
def cog_tile(
    z: int,
    x: int,
    y: int,
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12),
    rescale: Optional[str] = Query("auto", description="Example: 0,300 or auto"),
    colormap_name: str = Query("turbo"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    rescale_min, rescale_max = parse_rescale(
        rescale=rescale,
        dataset=dataset,
        cog_path=cog_path,
        bidx=resolved_bidx,
    )

    colormap = get_colormap(colormap_name)

    try:
        with COGReader(str(cog_path)) as cog:
            image = cog.tile(x, y, z, indexes=resolved_bidx)

            content = image.render(
                img_format="PNG",
                colormap=colormap,
                rescale=((rescale_min, rescale_max),),
            )

        return Response(content=content, media_type="image/png")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Cannot render tile.",
                "dataset": dataset,
                "month": month,
                "filename": cog_path.name,
                "band_index": resolved_bidx,
                "rescale": [rescale_min, rescale_max],
                "z": z,
                "x": x,
                "y": y,
                "error": str(exc),
            },
        )


@app.get("/cog/preview-url")
def preview_url(
    request: Request,
    dataset: str = Query(...),
    month: int = Query(..., ge=1, le=12),
    rescale: Optional[str] = Query("auto"),
    colormap_name: str = Query("turbo"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    rescale_min, rescale_max = parse_rescale(
        rescale=rescale,
        dataset=dataset,
        cog_path=cog_path,
        bidx=resolved_bidx,
    )
    rescale_text = format_rescale(rescale_min, rescale_max)

    return {
        "dataset": dataset,
        "month": month,
        "month_name": MONTH_NAMES[month],
        "filename": cog_path.name,
        "band_index": resolved_bidx,
        "rescale": [rescale_min, rescale_max],
        "rescale_text": rescale_text,
        "tile_url_template": build_tile_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
            bidx=resolved_bidx,
        ),
        "example_tile_url": build_example_tile_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
            bidx=resolved_bidx,
        ),
        "tilejson_url": build_tilejson_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
            bidx=resolved_bidx,
        ),
        "stats_url": (
            f"{get_public_base_url(request)}/cog/stats"
            f"?dataset={dataset}"
            f"&month={month}"
            f"&bidx={resolved_bidx}"
        ),
    }
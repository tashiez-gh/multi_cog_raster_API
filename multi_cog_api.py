import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rio_tiler.colormap import cmap
from rio_tiler.io import COGReader


# ============================================================
# Config
# ============================================================

APP_TITLE = "Multi COG Raster API"
APP_VERSION = "1.2.0"

BASE_DIR = Path(__file__).resolve().parent

COG_ROOT = Path(os.getenv("COG_ROOT", "data/output")).resolve()
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


# Classified temperature style.
# You can adjust these breaks later to match your QGIS QML exactly.
TEMPERATURE_CLASSES = [
    {
        "label": "< 28 °C",
        "min": None,
        "max": 28.0,
        "color": [49, 54, 149, 210],
    },
    {
        "label": "28–30 °C",
        "min": 28.0,
        "max": 30.0,
        "color": [69, 117, 180, 210],
    },
    {
        "label": "30–32 °C",
        "min": 30.0,
        "max": 32.0,
        "color": [116, 173, 209, 210],
    },
    {
        "label": "32–34 °C",
        "min": 32.0,
        "max": 34.0,
        "color": [171, 217, 233, 210],
    },
    {
        "label": "34–36 °C",
        "min": 34.0,
        "max": 36.0,
        "color": [255, 255, 191, 215],
    },
    {
        "label": "36–38 °C",
        "min": 36.0,
        "max": 38.0,
        "color": [253, 174, 97, 220],
    },
    {
        "label": "38–40 °C",
        "min": 38.0,
        "max": 40.0,
        "color": [244, 109, 67, 225],
    },
    {
        "label": "> 40 °C",
        "min": 40.0,
        "max": None,
        "color": [165, 0, 38, 230],
    },
]


RAINFALL_CLASSES = [
    {
        "label": "0 mm",
        "min": None,
        "max": 0.01,
        "color": [255, 255, 255, 0],
    },
    {
        "label": "0–20 mm",
        "min": 0.01,
        "max": 20.0,
        "color": [198, 219, 239, 190],
    },
    {
        "label": "20–50 mm",
        "min": 20.0,
        "max": 50.0,
        "color": [107, 174, 214, 200],
    },
    {
        "label": "50–100 mm",
        "min": 50.0,
        "max": 100.0,
        "color": [49, 130, 189, 210],
    },
    {
        "label": "100–150 mm",
        "min": 100.0,
        "max": 150.0,
        "color": [254, 217, 118, 220],
    },
    {
        "label": "150–200 mm",
        "min": 150.0,
        "max": 200.0,
        "color": [253, 141, 60, 225],
    },
    {
        "label": "> 200 mm",
        "min": 200.0,
        "max": None,
        "color": [189, 0, 38, 230],
    },
]


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


def resolve_style(dataset: str, style: Optional[str]) -> str:
    """
    style=auto:
      - temperature -> classified
      - rainfall -> continuous
    """
    config = get_dataset_config(dataset)
    variable = config["variable"]

    if style is None:
        style = "auto"

    style = style.lower().strip()

    if style not in ["auto", "continuous", "classified"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid style. Use style=auto, style=continuous, or style=classified.",
        )

    if style == "auto":
        if variable == "temperature":
            return "classified"
        return "continuous"

    return style


def get_classes_for_dataset(dataset: str) -> List[Dict]:
    config = get_dataset_config(dataset)
    variable = config["variable"]

    if variable == "temperature":
        return TEMPERATURE_CLASSES

    if variable == "rainfall":
        return RAINFALL_CLASSES

    return []


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
    validate_dataset(dataset)

    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month must be between 1 and 12.")

    tif_files = list_dataset_tifs(dataset)

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
# Classified renderer
# ============================================================

def get_tile_alpha_mask(image_data) -> Optional[np.ndarray]:
    """
    Try to extract rio-tiler mask.
    Usually 0 = transparent, 255 = valid.
    """
    mask = getattr(image_data, "mask", None)

    if mask is None:
        return None

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[0]

    return mask


def render_classified_png(image_data, classes: List[Dict]) -> bytes:
    """
    Render a single-band tile into RGBA PNG using fixed class breaks.
    """
    arr = image_data.data

    if arr.ndim == 3:
        arr = arr[0]

    arr = np.asarray(arr)

    h, w = arr.shape

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid_mask = np.isfinite(arr)

    alpha_mask = get_tile_alpha_mask(image_data)
    if alpha_mask is not None:
        valid_mask = valid_mask & (alpha_mask > 0)

    for item in classes:
        min_value = item["min"]
        max_value = item["max"]
        color = np.array(item["color"], dtype=np.uint8)

        class_mask = valid_mask.copy()

        if min_value is not None:
            class_mask = class_mask & (arr >= float(min_value))

        if max_value is not None:
            class_mask = class_mask & (arr < float(max_value))

        rgba[class_mask] = color

    img = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# Bounds helpers
# ============================================================

def get_cog_geographic_bounds(cog) -> Tuple[float, float, float, float]:
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
    style: str = "auto",
) -> str:
    base_url = get_public_base_url(request)

    url = (
        f"{base_url}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&style={style}"
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
    style: str = "auto",
) -> str:
    base_url = get_public_base_url(request)

    url = (
        f"{base_url}/cog/tiles/WebMercatorQuad/5/25/14.png"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&style={style}"
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
    style: str = "auto",
) -> str:
    base_url = get_public_base_url(request)

    url = (
        f"{base_url}/cog/tilejson.json"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&style={style}"
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
        "classified_temperature_example": (
            f"{base_url}/cog/tiles/WebMercatorQuad/5/25/14.png"
            f"?dataset=temp_ssp585_4160_p95_cog"
            f"&month=1"
            f"&style=classified"
            f"&rescale=auto"
            f"&colormap_name=turbo"
        ),
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

        default_style = resolve_style(dataset_name, "auto")

        items.append(
            {
                "name": dataset_name,
                "title": config["title"],
                "variable": config["variable"],
                "unit": config["unit"],
                "default_rescale": config["default_rescale"],
                "recommended_rescale_mode": "auto",
                "default_colormap": config["default_colormap"],
                "default_style": default_style,
                "classes": get_classes_for_dataset(dataset_name)
                if default_style == "classified"
                else [],
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
            style = resolve_style(dataset_name, "auto")

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
                    "default_style": style,
                    "default_colormap": colormap_name,
                    "classes": get_classes_for_dataset(dataset_name)
                    if style == "classified"
                    else [],
                    "tile_url_template": build_tile_url(
                        request=request,
                        dataset=dataset_name,
                        month=month_item["month"],
                        rescale=rescale,
                        colormap_name=colormap_name,
                        style=style,
                    ),
                    "example_tile_url": build_example_tile_url(
                        request=request,
                        dataset=dataset_name,
                        month=month_item["month"],
                        rescale=rescale,
                        colormap_name=colormap_name,
                        style=style,
                    ),
                    "tilejson_url": build_tilejson_url(
                        request=request,
                        dataset=dataset_name,
                        month=month_item["month"],
                        rescale=rescale,
                        colormap_name=colormap_name,
                        style=style,
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
    style = resolve_style(dataset, "auto")

    return {
        "dataset": dataset,
        "title": config["title"],
        "variable": config["variable"],
        "unit": config["unit"],
        "default_rescale": config["default_rescale"],
        "recommended_rescale_mode": "auto",
        "default_style": style,
        "default_colormap": config["default_colormap"],
        "classes": get_classes_for_dataset(dataset) if style == "classified" else [],
        "months": [
            {
                **month_item,
                "tile_url_template": build_tile_url(
                    request=request,
                    dataset=dataset,
                    month=month_item["month"],
                    rescale=rescale,
                    colormap_name=colormap_name,
                    style=style,
                )
                if month_item["available"]
                else None,
                "example_tile_url": build_example_tile_url(
                    request=request,
                    dataset=dataset,
                    month=month_item["month"],
                    rescale=rescale,
                    colormap_name=colormap_name,
                    style=style,
                )
                if month_item["available"]
                else None,
                "tilejson_url": build_tilejson_url(
                    request=request,
                    dataset=dataset,
                    month=month_item["month"],
                    rescale=rescale,
                    colormap_name=colormap_name,
                    style=style,
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
    style = resolve_style(dataset, "auto")

    return {
        "dataset": dataset,
        "title": config["title"],
        "variable": config["variable"],
        "unit": config["unit"],
        "month": month,
        "month_name": MONTH_NAMES[month],
        "filename": cog_path.name,
        "band_index": resolved_bidx,
        "default_style": style,
        "classes": get_classes_for_dataset(dataset) if style == "classified" else [],
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

    style = resolve_style(dataset, "auto")

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
        "default_style": style,
        "classes": get_classes_for_dataset(dataset) if style == "classified" else [],
        "stats": stats,
        "recommended_rescale": [rescale_min, rescale_max],
        "recommended_rescale_text": format_rescale(rescale_min, rescale_max),
    }


@app.get("/cog/tilejson.json")
def tilejson(
    request: Request,
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12),
    style: Optional[str] = Query("auto", description="auto, continuous, or classified"),
    rescale: Optional[str] = Query("auto", description="Example: 0,300 or auto"),
    colormap_name: str = Query("turbo"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    resolved_style = resolve_style(dataset, style)

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
        style=resolved_style,
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
        "style": resolved_style,
        "classes": get_classes_for_dataset(dataset) if resolved_style == "classified" else [],
    }


@app.get("/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png")
def cog_tile(
    z: int,
    x: int,
    y: int,
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12),
    style: Optional[str] = Query("auto", description="auto, continuous, or classified"),
    rescale: Optional[str] = Query("auto", description="Example: 0,300 or auto"),
    colormap_name: str = Query("turbo"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    resolved_style = resolve_style(dataset, style)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    rescale_min, rescale_max = parse_rescale(
        rescale=rescale,
        dataset=dataset,
        cog_path=cog_path,
        bidx=resolved_bidx,
    )

    try:
        with COGReader(str(cog_path)) as cog:
            image = cog.tile(x, y, z, indexes=resolved_bidx)

            if resolved_style == "classified":
                classes = get_classes_for_dataset(dataset)
                if not classes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No classified style configured for dataset={dataset}",
                    )

                content = render_classified_png(image, classes)

            else:
                colormap = get_colormap(colormap_name)

                content = image.render(
                    img_format="PNG",
                    colormap=colormap,
                    rescale=((rescale_min, rescale_max),),
                )

        return Response(content=content, media_type="image/png")

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Cannot render tile.",
                "dataset": dataset,
                "month": month,
                "filename": cog_path.name,
                "band_index": resolved_bidx,
                "style": resolved_style,
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
    style: Optional[str] = Query("auto", description="auto, continuous, or classified"),
    rescale: Optional[str] = Query("auto"),
    colormap_name: str = Query("turbo"),
    bidx: Optional[int] = Query(None, ge=1),
):
    validate_dataset(dataset)

    resolved_style = resolve_style(dataset, style)

    cog_path = find_month_cog(dataset, month)
    resolved_bidx = resolve_band_index(cog_path, month, requested_bidx=bidx)

    rescale_min, rescale_max = parse_rescale(
        rescale=rescale,
        dataset=dataset,
        cog_path=cog_path,
        bidx=resolved_bidx,
    )
    rescale_text = format_rescale(rescale_min, rescale_max)

    classes = get_classes_for_dataset(dataset) if resolved_style == "classified" else []

    return {
        "dataset": dataset,
        "month": month,
        "month_name": MONTH_NAMES[month],
        "filename": cog_path.name,
        "band_index": resolved_bidx,
        "style": resolved_style,
        "classes": classes,
        "rescale": [rescale_min, rescale_max],
        "rescale_text": rescale_text,
        "tile_url_template": build_tile_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
            bidx=resolved_bidx,
            style=resolved_style,
        ),
        "example_tile_url": build_example_tile_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
            bidx=resolved_bidx,
            style=resolved_style,
        ),
        "tilejson_url": build_tilejson_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
            bidx=resolved_bidx,
            style=resolved_style,
        ),
        "stats_url": (
            f"{get_public_base_url(request)}/cog/stats"
            f"?dataset={dataset}"
            f"&month={month}"
            f"&bidx={resolved_bidx}"
        ),
    }
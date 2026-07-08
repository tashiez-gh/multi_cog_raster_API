import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import morecantile
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rio_tiler.colormap import cmap
from rio_tiler.io import COGReader


# ============================================================
# Config
# ============================================================

APP_TITLE = "Multi COG Raster API"
APP_VERSION = "1.0.0"

# Render-safe path.
# Default expects:
# data/output/{dataset_name}/*.tif
COG_ROOT = Path(os.getenv("COG_ROOT", "data/output")).resolve()

# Optional. If empty, UI and API use current domain automatically.
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

BASE_DIR = Path(__file__).resolve().parent

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
# Helpers
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


def parse_rescale(rescale: Optional[str], dataset: str) -> Tuple[float, float]:
    default_min, default_max = SUPPORTED_DATASETS[dataset]["default_rescale"]

    if not rescale:
        return float(default_min), float(default_max)

    try:
        parts = [float(x.strip()) for x in rescale.split(",")]
        if len(parts) != 2:
            raise ValueError
        return parts[0], parts[1]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid rescale value. Use format: min,max. Example: rescale=0,300",
        )


def find_month_cog(dataset: str, month: int) -> Path:
    validate_dataset(dataset)

    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month must be between 1 and 12.")

    dataset_dir = COG_ROOT / dataset

    if not dataset_dir.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Dataset folder not found: {dataset_dir}",
                "hint": "Check your COG_ROOT environment variable and repository data/output folder.",
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

    month_patterns = [
        f"month_{month}",
        f"month{month}",
        f"m{month}",
        f"_{month}_",
        f"_{month:02d}_",
        f"-{month}-",
        f"-{month:02d}-",
        f"{month:02d}",
    ]

    # Prefer filename match.
    for tif in tif_files:
        stem = tif.stem.lower()
        if any(pattern in stem for pattern in month_patterns):
            return tif

    # Fallback: if there are exactly 12 files, use sorted order.
    if len(tif_files) >= 12:
        return tif_files[month - 1]

    raise HTTPException(
        status_code=404,
        detail={
            "message": f"Cannot find month {month} COG for dataset '{dataset}'.",
            "hint": "Use filenames containing month number, or keep exactly 12 sorted monthly COG files in each dataset folder.",
            "available_files": [f.name for f in tif_files],
        },
    )


def get_available_months(dataset: str) -> List[Dict]:
    validate_dataset(dataset)

    months = []

    for month in range(1, 13):
        try:
            path = find_month_cog(dataset, month)
            months.append(
                {
                    "month": month,
                    "name": MONTH_NAMES[month],
                    "available": True,
                    "filename": path.name,
                }
            )
        except HTTPException:
            months.append(
                {
                    "month": month,
                    "name": MONTH_NAMES[month],
                    "available": False,
                    "filename": None,
                }
            )

    return months


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


def build_tile_url(
    request: Request,
    dataset: str,
    month: int,
    rescale: str,
    colormap_name: str,
) -> str:
    base_url = get_public_base_url(request)
    return (
        f"{base_url}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&rescale={rescale}"
        f"&colormap_name={colormap_name}"
    )


def build_example_tile_url(
    request: Request,
    dataset: str,
    month: int,
    rescale: str,
    colormap_name: str,
) -> str:
    base_url = get_public_base_url(request)

    # Example tile around Thailand.
    return (
        f"{base_url}/cog/tiles/WebMercatorQuad/5/25/14.png"
        f"?dataset={dataset}"
        f"&month={month}"
        f"&rescale={rescale}"
        f"&colormap_name={colormap_name}"
    )


# ============================================================
# UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
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
        "example_tile": f"{base_url}/cog/tiles/WebMercatorQuad/5/25/14.png?dataset=rainfall_ssp245_cog&month=1&rescale=0,300&colormap_name=turbo",
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

            rescale_min, rescale_max = config["default_rescale"]
            rescale = f"{rescale_min},{rescale_max}"
            colormap_name = config["default_colormap"]

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
                    "default_rescale": config["default_rescale"],
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

    rescale_min, rescale_max = config["default_rescale"]
    rescale = f"{rescale_min},{rescale_max}"
    colormap_name = config["default_colormap"]

    return {
        "dataset": dataset,
        "title": config["title"],
        "variable": config["variable"],
        "unit": config["unit"],
        "default_rescale": config["default_rescale"],
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
            }
            for month_item in months
        ],
    }


# ============================================================
# COG endpoints
# ============================================================

@app.get("/cog/info")
def cog_info(
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12, description="Month number 1-12"),
):
    cog_path = find_month_cog(dataset, month)

    try:
        with COGReader(str(cog_path)) as cog:
            info = cog.info()
            bounds = cog.bounds
            geographic_bounds = cog.geographic_bounds
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
        "bounds": bounds,
        "geographic_bounds": geographic_bounds,
        "info": info,
    }


@app.get("/cog/tilejson.json")
def tilejson(
    request: Request,
    dataset: str = Query(..., description="Dataset name"),
    month: int = Query(..., ge=1, le=12),
    rescale: Optional[str] = Query(None, description="Example: 0,300"),
    colormap_name: str = Query("turbo"),
):
    validate_dataset(dataset)
    cog_path = find_month_cog(dataset, month)

    rescale_min, rescale_max = parse_rescale(rescale, dataset)
    rescale_text = f"{rescale_min},{rescale_max}"

    tile_url = build_tile_url(
        request=request,
        dataset=dataset,
        month=month,
        rescale=rescale_text,
        colormap_name=colormap_name,
    )

    try:
        with COGReader(str(cog_path)) as cog:
            bounds = cog.geographic_bounds
            minzoom = cog.minzoom
            maxzoom = cog.maxzoom
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
    rescale: Optional[str] = Query(None, description="Example: 0,300"),
    colormap_name: str = Query("turbo"),
    bidx: int = Query(1, ge=1, description="Band index. Default is 1."),
):
    cog_path = find_month_cog(dataset, month)
    rescale_min, rescale_max = parse_rescale(rescale, dataset)
    colormap = get_colormap(colormap_name)

    try:
        with COGReader(str(cog_path)) as cog:
            image = cog.tile(x, y, z, indexes=bidx)

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
    rescale: Optional[str] = Query(None),
    colormap_name: str = Query("turbo"),
):
    validate_dataset(dataset)

    rescale_min, rescale_max = parse_rescale(rescale, dataset)
    rescale_text = f"{rescale_min},{rescale_max}"

    return {
        "dataset": dataset,
        "month": month,
        "month_name": MONTH_NAMES[month],
        "tile_url_template": build_tile_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
        ),
        "example_tile_url": build_example_tile_url(
            request=request,
            dataset=dataset,
            month=month,
            rescale=rescale_text,
            colormap_name=colormap_name,
        ),
        "tilejson_url": (
            f"{get_public_base_url(request)}/cog/tilejson.json"
            f"?dataset={dataset}"
            f"&month={month}"
            f"&rescale={rescale_text}"
            f"&colormap_name={colormap_name}"
        ),
    }
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from titiler.core.factory import TilerFactory
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers


# ============================================================
# CONFIG
# ============================================================

# Change this path to the parent folder that contains your 4 COG dataset folders.
COG_ROOT = Path("/Users/tatar/Athentic/69/DCCE_69/script/output/")

API_BASE_URL = "http://localhost:8001"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Multi COG Raster API",
    description="COG catalog API + TiTiler tile API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ADD TITILER COG ENDPOINTS
# ============================================================

cog = TilerFactory(router_prefix="/cog")
app.include_router(cog.router, prefix="/cog", tags=["COG"])
add_exception_handlers(app, DEFAULT_STATUS_CODES)


# ============================================================
# HELPERS
# ============================================================

MONTH_LOOKUP = {
    "01_jan": "January",
    "02_feb": "February",
    "03_mar": "March",
    "04_apr": "April",
    "05_may": "May",
    "06_jun": "June",
    "07_jul": "July",
    "08_aug": "August",
    "09_sep": "September",
    "10_oct": "October",
    "11_nov": "November",
    "12_dec": "December",
}


def extract_month(filename: str) -> Optional[str]:
    lower_name = filename.lower()

    for key, month_name in MONTH_LOOKUP.items():
        if key in lower_name:
            return month_name

    return None


def guess_variable(filename: str, dataset_name: str) -> str:
    text = f"{filename} {dataset_name}".lower()

    if "pr_" in text or "rain" in text or "precip" in text:
        return "rainfall"

    if "tasmax" in text:
        return "tasmax"

    if "tasmin" in text:
        return "tasmin"

    if "temp" in text:
        return "temperature"

    return "raster"


def guess_rescale(variable: str) -> str:
    if variable == "rainfall":
        return "0,300"

    if variable == "tasmax":
        return "20,45"

    if variable == "tasmin":
        return "10,30"

    if variable == "temperature":
        return "15,45"

    return "0,1"


def guess_colormap(variable: str) -> str:
    if variable == "rainfall":
        return "turbo"

    if variable in ["tasmax", "tasmin", "temperature"]:
        return "turbo"

    return "viridis"


def build_tile_url(file_path: Path, rescale: str, colormap: str) -> str:
    file_url = f"file://{file_path}"

    return (
        f"{API_BASE_URL}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?url={file_url}"
        f"&bidx=1"
        f"&rescale={rescale}"
        f"&colormap_name={colormap}"
    )


def scan_layers() -> List[Dict]:
    if not COG_ROOT.exists():
        raise HTTPException(
            status_code=404,
            detail=f"COG_ROOT not found: {COG_ROOT}",
        )

    tif_files = sorted(list(COG_ROOT.rglob("*.tif")))

    layers = []

    for tif_path in tif_files:
        dataset_name = tif_path.parent.name
        filename = tif_path.name
        variable = guess_variable(filename, dataset_name)
        month = extract_month(filename)

        rescale = guess_rescale(variable)
        colormap = guess_colormap(variable)

        layer_id = f"{dataset_name}__{tif_path.stem}"

        layers.append(
            {
                "id": layer_id,
                "dataset": dataset_name,
                "name": tif_path.stem,
                "filename": filename,
                "variable": variable,
                "month": month,
                "path": str(tif_path),
                "url": f"file://{tif_path}",
                "tile_url": build_tile_url(tif_path, rescale, colormap),
                "info_url": f"{API_BASE_URL}/cog/info?url=file://{tif_path}",
                "preview_url": (
                    f"{API_BASE_URL}/cog/preview.png"
                    f"?url=file://{tif_path}"
                    f"&bidx=1"
                    f"&rescale={rescale}"
                    f"&colormap_name={colormap}"
                    f"&max_size=1024"
                ),
                "render": {
                    "bidx": 1,
                    "rescale": rescale,
                    "colormap_name": colormap,
                    "tile_size": 256,
                },
            }
        )

    return layers


# ============================================================
# CUSTOM CATALOG ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Multi COG API is running",
        "docs": f"{API_BASE_URL}/docs",
        "layers": f"{API_BASE_URL}/layers",
        "datasets": f"{API_BASE_URL}/datasets",
        "cog_info": f"{API_BASE_URL}/cog/info?url=file:///path/to/file.tif",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cog_root": str(COG_ROOT),
        "cog_root_exists": COG_ROOT.exists(),
    }


@app.get("/layers")
def list_layers():
    return scan_layers()


@app.get("/datasets")
def list_datasets():
    layers = scan_layers()

    dataset_names = sorted(set(layer["dataset"] for layer in layers))

    result = []

    for dataset_name in dataset_names:
        dataset_layers = [
            layer for layer in layers
            if layer["dataset"] == dataset_name
        ]

        result.append(
            {
                "dataset": dataset_name,
                "layer_count": len(dataset_layers),
                "variables": sorted(set(layer["variable"] for layer in dataset_layers)),
                "months": [
                    layer["month"]
                    for layer in dataset_layers
                    if layer["month"] is not None
                ],
                "layers_url": f"{API_BASE_URL}/datasets/{dataset_name}/layers",
            }
        )

    return result


@app.get("/datasets/{dataset_name}/layers")
def list_dataset_layers(dataset_name: str):
    layers = scan_layers()

    filtered = [
        layer for layer in layers
        if layer["dataset"] == dataset_name
    ]

    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset_name}",
        )

    return filtered


@app.get("/layers/{layer_id}")
def get_layer(layer_id: str):
    layers = scan_layers()

    for layer in layers:
        if layer["id"] == layer_id:
            return layer

    raise HTTPException(
        status_code=404,
        detail=f"Layer not found: {layer_id}",
    )
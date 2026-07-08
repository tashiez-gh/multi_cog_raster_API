"""
convert_tif_to_cog_and_api.py

Purpose
-------
1. Convert normal GeoTIFF files into Cloud Optimized GeoTIFFs, or COGs.
2. Provide a simple FastAPI API for listing COG layers and exposing TiTiler-compatible tile URLs.

Recommended install
-------------------
pip install rasterio rio-cogeo fastapi uvicorn pydantic

Optional, for serving real raster tiles from the same API:
pip install titiler.core titiler.application

Part A: Convert GeoTIFF to COG
------------------------------
python convert_tif_to_cog_and_api.py convert \
  --input-dir output/monthly_tif \
  --output-dir output/monthly_cog

Part B: Run simple API
----------------------
python convert_tif_to_cog_and_api.py api \
  --cog-dir output/monthly_cog \
  --host 0.0.0.0 \
  --port 8000

Then open:
http://localhost:8000/layers
http://localhost:8000/layers/{layer_id}

Tile URL pattern returned by this API:
http://localhost:8000/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=file:///absolute/path/to/file_cog.tif&bidx=1&rescale=min,max&colormap_name=turbo

Important
---------
This script returns TiTiler-style tile URLs. To make /cog/tiles/... work directly,
mount TiTiler in your FastAPI app or run a separate TiTiler server.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Optional

import rasterio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


class LayerInfo(BaseModel):
    id: str
    filename: str
    path: str
    crs: Optional[str]
    bounds: list[float]
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]
    tile_url: str
    file_url: str


def convert_one_tif_to_cog(input_tif: Path, output_tif: Path, overwrite: bool = False) -> Path:
    """Convert one GeoTIFF to COG."""
    if output_tif.exists() and not overwrite:
        print(f"Skip existing: {output_tif}")
        return output_tif

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    profile = cog_profiles.get("deflate")
    profile.update(
        {
            "blocksize": 512,
            "overview_resampling": "nearest",
        }
    )

    cog_translate(
        str(input_tif),
        str(output_tif),
        profile,
        in_memory=False,
        quiet=False,
    )
    print(f"COG wrote: {output_tif}")
    return output_tif


def convert_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> list[Path]:
    """Convert all .tif files in input_dir to COG."""
    tif_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif or .tiff files found in {input_dir}")

    written: list[Path] = []
    for tif in tif_files:
        out = output_dir / f"{tif.stem}_cog.tif"
        written.append(convert_one_tif_to_cog(tif, out, overwrite=overwrite))

    return written


def build_app(cog_dir: Path, public_base_url: str = "http://localhost:8001") -> FastAPI:
    """Build FastAPI app for COG layer catalog."""
    app = FastAPI(
        title="Monthly COG Raster API",
        description="Layer catalog API for monthly COG raster files.",
        version="1.0.0",
    )

    cog_dir = cog_dir.resolve()

    def get_cog_files() -> list[Path]:
        return sorted(list(cog_dir.glob("*.tif")) + list(cog_dir.glob("*.tiff")))

    def layer_id_from_path(path: Path) -> str:
        return path.stem

    def find_layer(layer_id: str) -> Path:
        for path in get_cog_files():
            if layer_id_from_path(path) == layer_id:
                return path
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_id}")

    def read_layer_info(path: Path, rescale: str = "0,1", colormap_name: str = "turbo") -> LayerInfo:
        with rasterio.open(path) as src:
            bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            file_url = f"{public_base_url}/files/{path.name}"

            # file:// is easiest for local demo. For production, use HTTPS object storage URL.
            local_file_url = f"file://{path}"
            tile_url = (
                f"{public_base_url}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
                f"?url={local_file_url}&bidx=1&rescale={rescale}&colormap_name={colormap_name}"
            )

            return LayerInfo(
                id=layer_id_from_path(path),
                filename=path.name,
                path=str(path),
                crs=str(src.crs) if src.crs else None,
                bounds=bounds,
                width=src.width,
                height=src.height,
                count=src.count,
                dtype=src.dtypes[0],
                nodata=src.nodata,
                tile_url=tile_url,
                file_url=file_url,
            )

    @app.get("/")
    def root():
        return {
            "message": "Monthly COG Raster API",
            "endpoints": ["/layers", "/layers/{layer_id}", "/files/{filename}"],
        }

    @app.get("/layers", response_model=list[LayerInfo])
    def list_layers(
        rescale: str = Query("0,1", description="Raster rescale range, for example 20,45."),
        colormap_name: str = Query("turbo", description="TiTiler colormap name."),
    ):
        files = get_cog_files()
        return [read_layer_info(path, rescale=rescale, colormap_name=colormap_name) for path in files]

    @app.get("/layers/{layer_id}", response_model=LayerInfo)
    def get_layer(
        layer_id: str,
        rescale: str = Query("0,1", description="Raster rescale range, for example 20,45."),
        colormap_name: str = Query("turbo", description="TiTiler colormap name."),
    ):
        path = find_layer(layer_id)
        return read_layer_info(path, rescale=rescale, colormap_name=colormap_name)

    @app.get("/files/{filename}")
    def serve_file(filename: str):
        path = cog_dir / filename
        if not path.exists() or path.suffix.lower() not in [".tif", ".tiff"]:
            raise HTTPException(status_code=404, detail="File not found")

        media_type = mimetypes.guess_type(path.name)[0] or "image/tiff"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.get("/catalog.json")
    def catalog_json():
        layers = [read_layer_info(path).model_dump() for path in get_cog_files()]
        return {"layers": layers}

    return app


def run_api(cog_dir: Path, host: str, port: int, public_base_url: str) -> None:
    import uvicorn

    app = build_app(cog_dir, public_base_url=public_base_url)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GeoTIFF to COG and run a simple COG layer API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert GeoTIFF files to COG.")
    convert_parser.add_argument("--input-dir", required=True, help="Input directory containing GeoTIFF files.")
    convert_parser.add_argument("--output-dir", required=True, help="Output directory for COG files.")
    convert_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing COG files.")

    api_parser = subparsers.add_parser("api", help="Run FastAPI layer catalog API.")
    api_parser.add_argument("--cog-dir", required=True, help="Directory containing COG files.")
    api_parser.add_argument("--host", default="0.0.0.0", help="API host. Default: 0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000, help="API port. Default: 8000")
    api_parser.add_argument(
        "--public-base-url",
        default="http://localhost:8001",
        help="Public base URL used when building file/tile URLs.",
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_directory(Path(args.input_dir), Path(args.output_dir), overwrite=args.overwrite)
    elif args.command == "api":
        run_api(Path(args.cog_dir), args.host, args.port, args.public_base_url)


if __name__ == "__main__":
    main()

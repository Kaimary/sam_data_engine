from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.engine import SamDataEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
engine = SamDataEngine(PROJECT_ROOT)
app = FastAPI(title="SAM Assisted Manual Data Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_shared_array_buffer_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return response


class AnnotationPayload(BaseModel):
    clicks: list[dict[str, Any]] = Field(default_factory=list)
    maskPngDataUrl: str
    color: str | None = None


class CompleteItemPayload(BaseModel):
    annotations: list[AnnotationPayload] = Field(default_factory=list)
    note: str | None = None


class SkipItemPayload(BaseModel):
    note: str | None = None


@app.on_event("startup")
def prepare_demo_assets() -> None:
    engine.ensure_quantized_model()


@app.get("/api/bootstrap")
def bootstrap(dataset: str = Query("diagram", pattern="^(diagram|plot)$")):
    return engine.bootstrap_payload(dataset)


@app.get("/api/items/next")
def next_item(dataset: str = Query("diagram", pattern="^(diagram|plot)$")):
    item = engine.next_pending(dataset)
    return {
        "dataset": dataset,
        "progress": engine.get_progress(dataset),
        "item": engine.serialize_item(item) if item else None,
    }


@app.get("/api/items/{item_id}/image")
def get_image(item_id: str, dataset: str = Query("diagram", pattern="^(diagram|plot)$")):
    try:
        item = engine.get_item(dataset, item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(item.image_path)


@app.get("/api/items/{item_id}/embedding.npy")
def get_embedding(item_id: str, dataset: str = Query("diagram", pattern="^(diagram|plot)$")):
    try:
        embedding_path = engine.ensure_embedding(dataset, item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to UI
        raise HTTPException(status_code=500, detail=f"Failed to prepare embedding: {exc}") from exc
    return FileResponse(embedding_path, media_type="application/octet-stream")


@app.post("/api/items/{item_id}/complete")
def complete_item(
    item_id: str,
    payload: CompleteItemPayload,
    dataset: str = Query("diagram", pattern="^(diagram|plot)$"),
):
    try:
        result = engine.save_annotations(
            dataset=dataset,
            item_id=item_id,
            annotations=[annotation.model_dump() for annotation in payload.annotations],
            status="completed",
            note=payload.note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    next_payload = next_item(dataset)
    return {"saved": result, **next_payload}


@app.post("/api/items/{item_id}/skip")
def skip_item(
    item_id: str,
    payload: SkipItemPayload,
    dataset: str = Query("diagram", pattern="^(diagram|plot)$"),
):
    try:
        result = engine.save_annotations(
            dataset=dataset,
            item_id=item_id,
            annotations=[],
            status="skipped",
            note=payload.note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    next_payload = next_item(dataset)
    return {"saved": result, **next_payload}


if engine.demo_dist.exists():
    app.mount("/", StaticFiles(directory=engine.demo_dist, html=True), name="frontend")
else:
    @app.get("/")
    def demo_not_built():
        return JSONResponse(
            {
                "message": "Frontend build is missing. Run `npm install && npm run build` in segment-anything/demo`."
            }
        )

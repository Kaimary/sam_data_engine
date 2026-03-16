from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DataItem:
    id: str
    dataset: str
    index: int
    total: int
    image_path: Path
    width: int
    height: int
    filename: str
    split: str | None
    title: str
    visual_intent: str | None
    method_section_title: str | None
    metadata_id: str | None


class SamDataEngine:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.segment_root = project_root / "segment-anything"
        self.demo_root = self._resolve_demo_root()
        self.demo_dist = self.demo_root / "dist"
        self.demo_model_root = self.demo_root / "model"
        self.data_root = project_root / "data"
        self.output_root = project_root / "outputs"
        self.embedding_root = self.output_root / "embeddings"
        self.auto_mask_root = self.output_root / "automatic_masks"
        self.state_root = self.output_root / "annotations"
        self.checkpoint_path = self._resolve_checkpoint()
        self.model_type = self._infer_model_type(self.checkpoint_path.name)
        self.quantized_model_path = self.demo_model_root / f"{self.model_type}_assisted_manual_quantized.onnx"
        self.float_model_path = self.output_root / "models" / f"{self.model_type}_assisted_manual.onnx"
        self.runtime_default = self._resolve_runtime_default()
        self._items = {
            "diagram": self._load_dataset("diagram"),
            "plot": self._load_dataset("plot"),
        }
        self._item_index = {
            dataset: {item.id: item for item in items}
            for dataset, items in self._items.items()
        }
        self._predictor = None
        self._automatic_mask_generator = None
        self._predictor_item: tuple[str, str] | None = None
        self._device = None
        self._embedding_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._prompt_lock = threading.Lock()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.embedding_root.mkdir(parents=True, exist_ok=True)
        self.auto_mask_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.float_model_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_demo_root(self) -> Path:
        local_demo_root = self.project_root / "demo"
        if local_demo_root.exists():
            return local_demo_root
        return self.segment_root / "demo"

    def _resolve_checkpoint(self) -> Path:
        checkpoint_dir = self.project_root / "checkpoint"
        candidates = sorted(checkpoint_dir.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        return candidates[0]

    @staticmethod
    def _resolve_runtime_default() -> Literal["server", "browser"]:
        raw_value = os.getenv("SAM_RUNTIME_DEFAULT", "server").strip().lower()
        if raw_value not in {"server", "browser"}:
            raise ValueError("SAM_RUNTIME_DEFAULT must be either 'server' or 'browser'")
        return cast(Literal["server", "browser"], raw_value)

    @staticmethod
    def _infer_model_type(filename: str) -> str:
        if "vit_h" in filename:
            return "vit_h"
        if "vit_l" in filename:
            return "vit_l"
        if "vit_b" in filename:
            return "vit_b"
        return "default"

    def _load_dataset(self, dataset: str) -> list[DataItem]:
        dataset_root = self.data_root / dataset
        image_root = dataset_root / "images"
        metadata_map: dict[str, dict[str, Any]] = {}
        for split_name in ("ref", "test"):
            manifest_path = dataset_root / f"{split_name}.json"
            if not manifest_path.exists():
                continue
            with manifest_path.open("r", encoding="utf-8") as handle:
                entries = json.load(handle)
            for entry in entries:
                rel_path = entry.get("path_to_gt_image")
                if not rel_path:
                    continue
                metadata_map[rel_path] = entry
        image_paths = sorted(path for path in image_root.iterdir() if path.is_file() and not path.name.startswith("."))
        items: list[DataItem] = []
        total = len(image_paths)
        for index, image_path in enumerate(image_paths):
            relative_key = f"images/{image_path.name}"
            metadata = metadata_map.get(relative_key, {})
            with Image.open(image_path) as image:
                width, height = image.size
            title = image_path.stem
            additional = metadata.get("additional_info") or {}
            items.append(
                DataItem(
                    id=f"{dataset}-{index:05d}",
                    dataset=dataset,
                    index=index,
                    total=total,
                    image_path=image_path,
                    width=width,
                    height=height,
                    filename=image_path.name,
                    split=(str(metadata.get("split")).replace(" ", "") if metadata.get("split") else None),
                    title=title,
                    visual_intent=metadata.get("visual_intent"),
                    method_section_title=additional.get("method_section_title"),
                    metadata_id=metadata.get("id"),
                )
            )
        return items

    def get_items(self, dataset: str) -> list[DataItem]:
        return self._items[dataset]

    def get_item(self, dataset: str, item_id: str) -> DataItem:
        try:
            return self._item_index[dataset][item_id]
        except KeyError as exc:
            raise KeyError(f"Unknown item {item_id} for dataset {dataset}") from exc

    def _dataset_state_dir(self, dataset: str) -> Path:
        path = self.state_root / dataset / "items"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dataset_mask_dir(self, dataset: str, item_id: str) -> Path:
        path = self.state_root / dataset / "masks" / item_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dataset_embedding_dir(self, dataset: str) -> Path:
        path = self.embedding_root / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dataset_auto_mask_dir(self, dataset: str, item_id: str) -> Path:
        path = self.auto_mask_root / dataset / item_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _auto_mask_manifest_path(self, dataset: str, item_id: str) -> Path:
        return self._dataset_auto_mask_dir(dataset, item_id) / "manifest.json"

    def _state_path(self, dataset: str, item_id: str) -> Path:
        return self._dataset_state_dir(dataset) / f"{item_id}.json"

    def get_status(self, dataset: str, item_id: str) -> dict[str, Any] | None:
        path = self._state_path(dataset, item_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_progress(self, dataset: str) -> dict[str, int]:
        completed = 0
        skipped = 0
        for item in self.get_items(dataset):
            state = self.get_status(dataset, item.id)
            if not state:
                continue
            status = state.get("status")
            if status == "completed":
                completed += 1
            elif status == "skipped":
                skipped += 1
        total = len(self.get_items(dataset))
        return {
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "pending": total - completed - skipped,
        }

    def next_pending(self, dataset: str) -> DataItem | None:
        for item in self.get_items(dataset):
            if not self._state_path(dataset, item.id).exists():
                return item
        return None

    def peek_next_pending(self, dataset: str, current_item_id: str) -> DataItem | None:
        for item in self.get_items(dataset):
            if item.id == current_item_id:
                continue
            if not self._state_path(dataset, item.id).exists():
                return item
        return None

    def serialize_item(self, item: DataItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "dataset": item.dataset,
            "index": item.index,
            "total": item.total,
            "filename": item.filename,
            "width": item.width,
            "height": item.height,
            "title": item.title,
            "split": item.split,
            "visualIntent": item.visual_intent,
            "methodSectionTitle": item.method_section_title,
            "metadataId": item.metadata_id,
            "imageUrl": f"/api/items/{item.id}/image?dataset={item.dataset}",
            "embeddingUrl": f"/api/items/{item.id}/embedding.npy?dataset={item.dataset}",
        }

    def bootstrap_payload(self, dataset: str) -> dict[str, Any]:
        item = self.next_pending(dataset)
        return {
            "dataset": dataset,
            "progress": self.get_progress(dataset),
            "item": self.serialize_item(item) if item else None,
        }

    def ensure_quantized_model(self) -> Path:
        with self._model_lock:
            if self.quantized_model_path.exists():
                return self.quantized_model_path
            self.demo_model_root.mkdir(parents=True, exist_ok=True)
            sys.path.insert(0, str(self.segment_root))
            from scripts.export_onnx_model import run_export

            print(f"[engine] exporting ONNX decoder to {self.float_model_path}")
            run_export(
                model_type=self.model_type,
                checkpoint=str(self.checkpoint_path),
                output=str(self.float_model_path),
                opset=17,
                return_single_mask=True,
                gelu_approximate=True,
                use_stability_score=False,
                return_extra_metrics=False,
            )
            try:
                from onnxruntime.quantization import QuantType
                from onnxruntime.quantization.quantize import quantize_dynamic
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required to quantize the ONNX model") from exc

            print(f"[engine] quantizing ONNX decoder to {self.quantized_model_path}")
            quantize_dynamic(
                model_input=str(self.float_model_path),
                model_output=str(self.quantized_model_path),
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
            )
            return self.quantized_model_path

    def _get_device(self) -> str:
        if self._device is not None:
            return self._device
        import torch

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        return self._device

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        sys.path.insert(0, str(self.segment_root))
        from segment_anything import SamPredictor, sam_model_registry

        device = self._get_device()
        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        sam.to(device=device)
        sam.eval()
        predictor = SamPredictor(sam)
        self._predictor = predictor
        print(f"[engine] loaded SAM image encoder on {device}")
        return predictor

    def _load_automatic_mask_generator(self):
        if self._automatic_mask_generator is not None:
            return self._automatic_mask_generator

        predictor = self._load_predictor()
        sys.path.insert(0, str(self.segment_root))
        from segment_anything import SamAutomaticMaskGenerator

        self._automatic_mask_generator = SamAutomaticMaskGenerator(
            predictor.model,
            points_per_side=16,
            points_per_batch=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            box_nms_thresh=0.7,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=0,
            output_mode="binary_mask",
        )
        return self._automatic_mask_generator

    def ensure_embedding(self, dataset: str, item_id: str) -> Path:
        item = self.get_item(dataset, item_id)
        embedding_path = self._dataset_embedding_dir(dataset) / f"{item.id}.npy"
        if embedding_path.exists():
            return embedding_path
        with self._embedding_lock:
            if embedding_path.exists():
                return embedding_path
            predictor = self._load_predictor()
            image = np.array(Image.open(item.image_path).convert("RGB"))
            predictor.set_image(image)
            embedding = predictor.get_image_embedding().detach().cpu().numpy()
            np.save(embedding_path, embedding)
            print(f"[engine] cached embedding for {item.id} at {embedding_path}")
        return embedding_path

    def runtime_payload(self, mode: Literal["server", "browser"] | None = None) -> dict[str, str]:
        active_mode = mode or self.runtime_default
        if active_mode == "browser":
            return {
                "mode": "browser",
                "defaultMode": self.runtime_default,
                "location": "browser",
                "device": "webgpu",
                "label": "Browser WebGPU",
                "interactiveLabel": "Browser WebGPU",
                "automaticMaskLabel": "Server",
                "modelUrl": "/api/runtime/browser-model.onnx",
            }
        device = self._get_device()
        return {
            "mode": "server",
            "defaultMode": self.runtime_default,
            "location": "server",
            "device": device,
            "label": f"Server {device.upper()}",
            "interactiveLabel": f"Server {device.upper()}",
            "automaticMaskLabel": f"Server {device.upper()}",
            "modelUrl": "/api/runtime/browser-model.onnx",
        }

    def _ensure_predictor_state(self, dataset: str, item_id: str):
        item = self.get_item(dataset, item_id)
        predictor = self._load_predictor()
        state_key = (dataset, item_id)
        if self._predictor_item == state_key and predictor.is_image_set:
            return predictor, item

        embedding_path = self.ensure_embedding(dataset, item_id)
        embedding = np.load(embedding_path)

        import torch
        from segment_anything.utils.transforms import ResizeLongestSide

        features = torch.from_numpy(embedding).to(device=self._get_device())
        input_size = ResizeLongestSide.get_preprocess_shape(
            item.height, item.width, predictor.model.image_encoder.img_size
        )

        predictor.reset_image()
        predictor.features = features
        predictor.original_size = (item.height, item.width)
        predictor.input_size = input_size
        predictor.is_image_set = True
        self._predictor_item = state_key
        return predictor, item

    @staticmethod
    def _mask_bbox(mask_array: np.ndarray) -> list[int]:
        ys, xs = np.where(mask_array)
        if ys.size == 0 or xs.size == 0:
            return [0, 0, 0, 0]
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max())
        y1 = int(ys.max())
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

    @staticmethod
    def _png_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    @staticmethod
    def _hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
        normalized = color.lstrip("#")
        if len(normalized) != 6:
            raise ValueError(f"Expected RGB hex color, got: {color}")
        return (
            int(normalized[0:2], 16),
            int(normalized[2:4], 16),
            int(normalized[4:6], 16),
            alpha,
        )

    def _mask_layer_payload(self, mask_array: np.ndarray, color: str) -> dict[str, Any]:
        binary = mask_array.astype(np.uint8)
        height, width = binary.shape
        overlay_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        overlay_rgba[binary > 0] = self._hex_to_rgba(color, 180)

        mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        mask_rgba[binary > 0] = (255, 255, 255, 255)

        area = int(binary.sum())
        return {
            "overlayUrl": self._png_data_url(Image.fromarray(overlay_rgba, mode="RGBA")),
            "maskPngDataUrl": self._png_data_url(Image.fromarray(mask_rgba, mode="RGBA")),
            "area": area,
            "bbox": self._mask_bbox(binary > 0),
        }

    @staticmethod
    def _automatic_mask_palette() -> list[str]:
        return ["#ff7a59", "#5abf90", "#4fa3ff", "#f6bd60", "#9d7bff", "#f28482"]

    def _filter_automatic_masks(
        self,
        item: DataItem,
        generated_masks: list[dict[str, Any]],
        limit: int = 32,
    ) -> list[dict[str, Any]]:
        image_area = item.width * item.height
        min_area = max(64, int(image_area * 0.001))
        max_area = int(image_area * 0.98)
        filtered = [
            mask
            for mask in generated_masks
            if min_area <= int(mask.get("area", 0)) <= max_area
        ]
        if not filtered:
            filtered = [mask for mask in generated_masks if int(mask.get("area", 0)) > 0]

        filtered.sort(
            key=lambda mask: (
                int(mask.get("area", 0)),
                float(mask.get("predicted_iou", 0.0)),
                float(mask.get("stability_score", 0.0)),
            ),
            reverse=True,
        )
        return filtered[:limit]

    def _automatic_mask_records_to_payload(
        self,
        dataset: str,
        item_id: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for record in records:
            mask_path = self.project_root / record["maskPath"]
            with Image.open(mask_path) as image:
                mask_array = np.array(image.convert("L")) > 0
            payload = self._mask_layer_payload(mask_array.astype(np.uint8), record["color"])
            payloads.append(
                {
                    "id": record["id"],
                    "itemId": item_id,
                    "dataset": dataset,
                    "color": record["color"],
                    "clicks": [],
                    "source": "automatic",
                    "score": record.get("score"),
                    **payload,
                }
            )
        return payloads

    def automatic_masks(self, dataset: str, item_id: str) -> dict[str, Any]:
        item = self.get_item(dataset, item_id)
        manifest_path = self._auto_mask_manifest_path(dataset, item_id)
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            return {
                "dataset": dataset,
                "itemId": item_id,
                "masks": self._automatic_mask_records_to_payload(dataset, item_id, records),
            }

        with self._prompt_lock:
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    records = json.load(handle)
                return {
                    "dataset": dataset,
                    "itemId": item_id,
                    "masks": self._automatic_mask_records_to_payload(dataset, item_id, records),
                }

            generator = self._load_automatic_mask_generator()
            image = np.array(Image.open(item.image_path).convert("RGB"))
            generated_masks = generator.generate(image)
            filtered_masks = self._filter_automatic_masks(item, generated_masks)
            palette = self._automatic_mask_palette()
            mask_dir = self._dataset_auto_mask_dir(dataset, item_id)
            records: list[dict[str, Any]] = []

            for index, generated_mask in enumerate(filtered_masks, start=1):
                mask_array = generated_mask["segmentation"].astype(np.uint8)
                color = palette[(index - 1) % len(palette)]
                payload = self._mask_layer_payload(mask_array, color)
                mask_bytes = self.decode_data_url(payload["maskPngDataUrl"])
                mask_path = mask_dir / f"mask_{index:03d}.png"
                mask_path.write_bytes(mask_bytes)
                records.append(
                    {
                        "id": f"{item_id}-auto-{index:03d}",
                        "maskPath": str(mask_path.relative_to(self.project_root)),
                        "color": color,
                        "score": float(generated_mask.get("predicted_iou", 0.0)),
                        "area": payload["area"],
                        "bbox": payload["bbox"],
                    }
                )

            manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "dataset": dataset,
            "itemId": item_id,
            "masks": self._automatic_mask_records_to_payload(dataset, item_id, records),
        }

    def predict_mask(
        self,
        dataset: str,
        item_id: str,
        clicks: list[dict[str, Any]],
        color: str = "#2e86ab",
    ) -> dict[str, Any]:
        if not clicks:
            raise ValueError("At least one click is required for prediction.")

        with self._prompt_lock:
            predictor, item = self._ensure_predictor_state(dataset, item_id)
            point_clicks = [click for click in clicks if int(click["clickType"]) in (0, 1)]
            box_clicks = [click for click in clicks if int(click["clickType"]) in (2, 3)]

            point_coords = (
                np.array([[float(click["x"]), float(click["y"])] for click in point_clicks], dtype=np.float32)
                if point_clicks
                else None
            )
            point_labels = (
                np.array([int(click["clickType"]) for click in point_clicks], dtype=np.int32)
                if point_clicks
                else None
            )
            box = None
            if box_clicks:
                top_left = next((click for click in box_clicks if int(click["clickType"]) == 2), None)
                bottom_right = next((click for click in box_clicks if int(click["clickType"]) == 3), None)
                if top_left is None or bottom_right is None:
                    raise ValueError("Box prompt requires both top-left and bottom-right corners.")
                box = np.array(
                    [
                        float(top_left["x"]),
                        float(top_left["y"]),
                        float(bottom_right["x"]),
                        float(bottom_right["y"]),
                    ],
                    dtype=np.float32,
                )

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
                return_logits=False,
            )

        mask = masks[0].astype(np.uint8)
        payload = self._mask_layer_payload(mask, color)
        return {
            "itemId": item.id,
            "dataset": dataset,
            "color": color,
            "score": float(scores[0]),
            "width": item.width,
            "height": item.height,
            **payload,
        }

    @staticmethod
    def decode_data_url(data_url: str) -> bytes:
        header, encoded = data_url.split(",", 1)
        if not header.startswith("data:image/png;base64"):
            raise ValueError("Only PNG mask uploads are supported")
        return base64.b64decode(encoded)

    @staticmethod
    def mask_summary(mask_bytes: bytes) -> tuple[int, list[int]]:
        with Image.open(io.BytesIO(mask_bytes)) as image:
            mask_array = np.array(image.convert("L")) > 0
        ys, xs = np.where(mask_array)
        area = int(mask_array.sum())
        if area == 0:
            bbox = [0, 0, 0, 0]
        else:
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max())
            y1 = int(ys.max())
            bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        return area, bbox

    def save_annotations(
        self,
        dataset: str,
        item_id: str,
        annotations: list[dict[str, Any]],
        status: Literal["completed", "skipped"],
        note: str | None = None,
    ) -> dict[str, Any]:
        item = self.get_item(dataset, item_id)
        state_path = self._state_path(dataset, item_id)
        mask_dir = self._dataset_mask_dir(dataset, item_id)
        records: list[dict[str, Any]] = []
        for index, annotation in enumerate(annotations, start=1):
            mask_bytes = self.decode_data_url(annotation["maskPngDataUrl"])
            mask_path = mask_dir / f"mask_{index:03d}.png"
            mask_path.write_bytes(mask_bytes)
            area, bbox = self.mask_summary(mask_bytes)
            clicks = annotation.get("clicks") or []
            records.append(
                {
                    "index": index,
                    "maskPath": str(mask_path.relative_to(self.project_root)),
                    "area": area,
                    "bbox": bbox,
                    "clicks": clicks,
                    "positiveClicks": sum(1 for click in clicks if click.get("clickType") == 1),
                    "negativeClicks": sum(1 for click in clicks if click.get("clickType") == 0),
                    "color": annotation.get("color"),
                    "source": annotation.get("source") or "manual",
                    "score": annotation.get("score"),
                }
            )

        payload = {
            "status": status,
            "dataset": dataset,
            "item": {
                "id": item.id,
                "title": item.title,
                "filename": item.filename,
                "imagePath": str(item.image_path.relative_to(self.project_root)),
                "width": item.width,
                "height": item.height,
                "split": item.split,
                "visualIntent": item.visual_intent,
                "methodSectionTitle": item.method_section_title,
                "metadataId": item.metadata_id,
            },
            "annotationCount": len(records),
            "annotations": records,
            "note": note,
        }
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

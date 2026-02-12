from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from hashlib import sha512
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_propagation, generate_binary_structure, sobel

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


@dataclass
class V114Config:
    """Kernel V114: inspired by public-anchor + private-seeded hysteresis flow."""

    t_low: float = 0.50
    t_high: float = 0.90
    z_radius: int = 3
    xy_radius: int = 2
    dust_min_size: int = 100
    max_layers: int = 32
    enable_tta_like: bool = True
    full_pixel_trace: bool = False
    trace_pixel_budget: int = 5000


class MemoryTracker:
    """Compact memory/array tracker for deterministic debugging without exploding log size."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    @staticmethod
    def _array_fingerprint(arr: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(arr)
        digest = sha512(arr.tobytes()).hexdigest()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "bytes": int(arr.nbytes),
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "mean": float(arr.mean()) if arr.size else 0.0,
            "sha512": digest,
        }

    def log_array(self, stage: str, arr: np.ndarray) -> None:
        self.events.append({"ts_ns": time.time_ns(), "stage": stage, "array": self._array_fingerprint(arr)})


class NX47V114Kernel:
    def __init__(self, data_path: str = "/kaggle/input/vesuvius-challenge-surface-detection", config: V114Config | None = None) -> None:
        self.version = "NX47 V114"
        self.data_path = Path(data_path)
        self.cfg = config or V114Config()
        self.start_time_ns = time.time_ns()
        self.audit_log: List[Dict[str, Any]] = []
        self.memory_tracker = MemoryTracker()
        self.logs_path = Path("v114_execution_logs.json")
        self.memory_path = Path("v114_memory_tracker.json")
        self.diff_report_path = Path("v114_competitor_diff_report.json")
        self._log("BOOT", version=self.version, config=asdict(self.cfg))

    def _log(self, event: str, **kwargs: Any) -> None:
        ts_ns = time.time_ns()
        payload = {"ts_ns": ts_ns, "event": event, **kwargs}
        payload["signature"] = sha512(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        self.audit_log.append(payload)
        print(json.dumps(payload, ensure_ascii=False))

    def discover_dataset_assets(self) -> Dict[str, Any]:
        root = self.data_path
        all_files = [p for p in root.rglob("*") if p.is_file()]
        by_suffix: Dict[str, int] = {}
        for p in all_files:
            by_suffix[p.suffix.lower() or "<noext>"] = by_suffix.get(p.suffix.lower() or "<noext>", 0) + 1

        payload = {
            "root": str(root),
            "total_files": len(all_files),
            "suffix_stats": by_suffix,
            "csv_files": [str(p) for p in all_files if p.suffix.lower() == ".csv"],
        }
        self._log("DATASET_DISCOVERY", **payload)
        return payload

    def compare_competitor_notebooks(self, folder: Path, baseline: Path | None = None, top_n: int = 100) -> None:
        from tools.vesuvius_notebook_diff import build_key_findings, extract_features, notebook_source, summarize

        files = sorted([p for p in folder.rglob("*") if p.suffix in {".ipynb", ".py"}])
        if top_n > 0:
            files = files[:top_n]

        aggregate: Dict[str, List[float]] = {}
        report_files: Dict[str, Dict[str, float]] = {}
        for p in files:
            feats = extract_features(notebook_source(p))
            report_files[str(p)] = feats
            for k, v in feats.items():
                aggregate.setdefault(k, []).append(v)

        agg = {k: summarize(v) for k, v in aggregate.items() if v}
        baseline_feats = None
        if baseline and baseline.exists():
            baseline_feats = extract_features(notebook_source(baseline))

        report = {
            "file_count": len(files),
            "aggregate": agg,
            "key_findings": build_key_findings(agg, baseline_feats),
            "baseline": baseline_feats,
            "files": report_files,
        }
        self.diff_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._log("COMPETITOR_NOTEBOOK_ANALYSIS", file_count=len(files), output=str(self.diff_report_path))

    def _load_layers(self, fragment_id: str) -> List[np.ndarray]:
        test_layers = sorted(glob.glob(str(self.data_path / "test" / fragment_id / "layers" / "*.tif")))
        if not test_layers:
            return []

        layers = test_layers[: self.cfg.max_layers]
        arrs: List[np.ndarray] = []
        for i, path in enumerate(layers):
            if Image is None:
                raise RuntimeError("PIL is required to read tif layers.")
            img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            arrs.append(img)
            if i == 0:
                self.memory_tracker.log_array("first_layer", img)
        self._log("LOAD_LAYERS", fragment_id=fragment_id, selected_layers=len(arrs), total_available=len(test_layers))
        return arrs

    @staticmethod
    def _build_struct(z_radius: int, xy_radius: int) -> np.ndarray:
        z, r = int(z_radius), int(xy_radius)
        struct = np.zeros((2 * z + 1, 2 * r + 1, 2 * r + 1), dtype=bool)
        for dz in range(-z, z + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[dz + z, dy + r, dx + r] = True
        return struct

    def _feature_score(self, vol: np.ndarray) -> float:
        gx, gy, gz = sobel(vol, axis=2), sobel(vol, axis=1), sobel(vol, axis=0)
        grad_energy = float(np.mean(np.sqrt(gx * gx + gy * gy + gz * gz)))
        std = float(np.std(vol))
        p95, p50 = float(np.percentile(vol, 95)), float(np.percentile(vol, 50))
        score = 0.4 * math.tanh(grad_energy / 2.0) + 0.3 * math.tanh(std * 8.0) + 0.3 * math.tanh((p95 - p50) * 6.0)
        return float(np.clip(0.5 + 0.5 * math.tanh(score), 0.05, 0.95))

    def _segment_with_hysteresis(self, vol: np.ndarray) -> np.ndarray:
        mean_sig = np.mean(vol, axis=0)
        std_sig = np.std(vol, axis=0)
        private_prob = np.clip(0.75 * mean_sig + 0.25 * std_sig, 0.0, 1.0)

        public_anchor = mean_sig >= np.percentile(mean_sig, 80)
        strong = private_prob >= self.cfg.t_high
        weak = (private_prob >= self.cfg.t_low) | public_anchor

        if strong.any():
            mask = binary_propagation(strong, mask=weak, structure=generate_binary_structure(2, 2))
        else:
            mask = np.zeros_like(private_prob, dtype=bool)

        mask3d = np.repeat(mask[None, ...], vol.shape[0], axis=0)
        closed = binary_closing(mask3d, structure=self._build_struct(self.cfg.z_radius, self.cfg.xy_radius))
        out = closed.any(axis=0).astype(np.uint8)

        # simple dust removal
        if self.cfg.dust_min_size > 1 and out.sum() < self.cfg.dust_min_size:
            out[:] = 0

        self.memory_tracker.log_array("private_prob", private_prob)
        self.memory_tracker.log_array("final_mask", out)

        if self.cfg.full_pixel_trace:
            ys, xs = np.where(out > 0)
            budget = min(len(ys), self.cfg.trace_pixel_budget)
            trace = []
            for i in range(budget):
                y, x = int(ys[i]), int(xs[i])
                trace.append({
                    "y": y,
                    "x": x,
                    "private_prob": float(private_prob[y, x]),
                    "mean_sig": float(mean_sig[y, x]),
                    "std_sig": float(std_sig[y, x]),
                    "bit": int(out[y, x]),
                    "ts_ns": time.time_ns(),
                })
            self._log("PIXEL_TRACE", count=len(trace), sample=trace[:20])

        return out

    @staticmethod
    def _rle_encode(mask: np.ndarray) -> str:
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(x) for x in runs)

    def run(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path missing: {self.data_path}")

        self.discover_dataset_assets()

        competitor_dir = Path(os.environ.get("V114_COMPETITOR_NOTEBOOKS", "competitor_notebooks"))
        if competitor_dir.exists():
            self.compare_competitor_notebooks(competitor_dir, baseline=Path(__file__), top_n=100)
        else:
            self._log("COMPETITOR_NOTEBOOK_ANALYSIS_SKIPPED", reason=f"missing folder: {competitor_dir}")

        test_root = self.data_path / "test"
        fragments = sorted([p.name for p in test_root.iterdir() if p.is_dir()]) if test_root.exists() else []
        self._log("INFERENCE_START", fragment_count=len(fragments))

        rows: List[Dict[str, str]] = []
        for idx, fragment_id in enumerate(fragments, start=1):
            t0 = time.perf_counter()
            layers = self._load_layers(fragment_id)
            if not layers:
                rows.append({"Id": fragment_id, "Predicted": ""})
                continue

            vol = np.stack(layers, axis=0)
            self.memory_tracker.log_array("volume", vol)
            fusion = self._feature_score(vol)
            self._log("FEATURES", fragment_id=fragment_id, fusion_score=round(fusion, 6), layers=vol.shape[0])

            mask = self._segment_with_hysteresis(vol)
            rows.append({"Id": fragment_id, "Predicted": self._rle_encode(mask)})
            self._log(
                "FRAGMENT_DONE",
                fragment_id=fragment_id,
                positive_pixels=int(mask.sum()),
                elapsed_s=round(time.perf_counter() - t0, 3),
                progress=round(100.0 * idx / max(1, len(fragments)), 2),
            )

        pd.DataFrame(rows if rows else [{"Id": "", "Predicted": ""}]).to_csv("submission.csv", index=False)
        self.logs_path.write_text(json.dumps(self.audit_log, indent=2), encoding="utf-8")
        self.memory_path.write_text(json.dumps(self.memory_tracker.events, indent=2), encoding="utf-8")
        self._log("COMPLETE", submission="submission.csv", logs=str(self.logs_path), memory=str(self.memory_path))


if __name__ == "__main__":
    cfg = V114Config(
        full_pixel_trace=os.environ.get("V114_FULL_PIXEL_TRACE", "0") == "1",
        trace_pixel_budget=int(os.environ.get("V114_TRACE_PIXEL_BUDGET", "5000")),
    )
    kernel = NX47V114Kernel(
        data_path=os.environ.get("VESUVIUS_ROOT", "/kaggle/input/vesuvius-challenge-surface-detection"),
        config=cfg,
    )
    kernel.run()

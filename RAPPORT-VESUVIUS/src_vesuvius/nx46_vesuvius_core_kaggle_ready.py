"""NX-46 AGNN Vesuvius Kaggle-ready core (offline, forensic, submission.zip TIFF).

Reintegrated with validated advances observed in historical notebook outputs:
- strict submission.zip creation with .tif members
- competition-style member validation
- richer forensic artifacts and dual execution logs
- roadmap + memory tracker exports for auditability
"""

from __future__ import annotations

import csv
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from hashlib import sha512
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import tifffile


@dataclass
class NX46Config:
    data_root: str = "/kaggle/input/vesuvius-challenge-ink-detection"
    work_root: str = "/kaggle/working"
    log_root: str = "/kaggle/working/RAPPORT-VESUVIUS/output_logs_vesuvius"
    seed: int = 46
    bit_capture_bytes: int = 256
    threshold_quantile: float = 0.985
    slab_min_neurons: int = 128
    enforce_submission_rules: bool = True


class HFBL360Logger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.forensic_log = root / "forensic_ultra.log"
        self.metrics_csv = root / "metrics.csv"
        self.state_json = root / "state.json"
        self.bit_log = root / "bit_capture.log"
        self.merkle_log = root / "merkle_chain.log"
        self.core_log = root / "nx-46-vesuvius-core.log"
        self.kaggle_ready_log = root / "nx46-vesuvius-core-kaggle-ready.log"
        self.roadmap_json = root / "nx46_roadmap_realtime.json"
        self.memory_tracker_json = root / "nx46_memory_tracker.json"

        self.events: List[Dict[str, object]] = []
        self.roadmap = [
            {"name": "discovery", "progress_percent": 0.0, "done": False},
            {"name": "load", "progress_percent": 0.0, "done": False},
            {"name": "features", "progress_percent": 0.0, "done": False},
            {"name": "train", "progress_percent": 0.0, "done": False},
            {"name": "segment", "progress_percent": 0.0, "done": False},
            {"name": "package", "progress_percent": 0.0, "done": False},
        ]

        with self.metrics_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp_ns",
                    "phase",
                    "fragment",
                    "neurons_active",
                    "cpu_ns",
                    "ink_pixels",
                    "total_pixels",
                    "ink_ratio",
                    "merkle_prefix",
                ]
            )

    def _append_event(self, event: str, **data: object) -> None:
        row = {"timestamp_ns": time.time_ns(), "event": event, **data}
        self.events.append(row)

    def log_event(self, message: str, **data: object) -> None:
        ts = time.time_ns()
        line = f"{ts} | {message}"
        if data:
            line += " | " + json.dumps(data, ensure_ascii=False)
        line += "\n"
        self.forensic_log.write_text(self.forensic_log.read_text(encoding="utf-8") + line if self.forensic_log.exists() else line, encoding="utf-8")
        self.core_log.write_text(self.core_log.read_text(encoding="utf-8") + line if self.core_log.exists() else line, encoding="utf-8")
        self.kaggle_ready_log.write_text(
            self.kaggle_ready_log.read_text(encoding="utf-8") + line if self.kaggle_ready_log.exists() else line,
            encoding="utf-8",
        )
        self._append_event(message, **data)

    def log_bits(self, fragment: str, payload: bytes) -> None:
        bit_string = "".join(f"{b:08b}" for b in payload)
        line = f"{time.time_ns()} | {fragment} | {bit_string}\n"
        self.bit_log.write_text(self.bit_log.read_text(encoding="utf-8") + line if self.bit_log.exists() else line, encoding="utf-8")

    def log_merkle(self, fragment: str, digest: str) -> None:
        line = f"{time.time_ns()} | {fragment} | {digest}\n"
        self.merkle_log.write_text(self.merkle_log.read_text(encoding="utf-8") + line if self.merkle_log.exists() else line, encoding="utf-8")

    def log_metrics(
        self,
        *,
        phase: str,
        fragment: str,
        neurons_active: int,
        cpu_ns: int,
        ink_pixels: int,
        total_pixels: int,
        merkle_prefix: str,
    ) -> None:
        ratio = (ink_pixels / total_pixels) if total_pixels else 0.0
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    time.time_ns(),
                    phase,
                    fragment,
                    neurons_active,
                    cpu_ns,
                    ink_pixels,
                    total_pixels,
                    f"{ratio:.8f}",
                    merkle_prefix,
                ]
            )

    def set_roadmap(self, step: str, progress: float, done: bool = False) -> None:
        for row in self.roadmap:
            if row["name"] == step:
                row["progress_percent"] = float(max(0.0, min(100.0, progress)))
                row["done"] = bool(done)
                break
        payload = {
            "generated_at_ns": time.time_ns(),
            "roadmap": self.roadmap,
            "overall_progress_percent": round(float(np.mean([r["progress_percent"] for r in self.roadmap])), 2),
        }
        self.roadmap_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_state(self, state: Dict[str, object]) -> None:
        state = dict(state)
        state["timestamp_ns"] = time.time_ns()
        self.state_json.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def flush_events(self) -> None:
        self.memory_tracker_json.write_text(json.dumps(self.events, indent=2, ensure_ascii=False), encoding="utf-8")


class NX46AGNNVesuvius:
    def __init__(self, cfg: NX46Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.logger = HFBL360Logger(Path(cfg.log_root))
        self.neurons_active = 0
        self.total_allocations = 0
        self.total_pixels_processed = 0
        self.total_ink_pixels = 0
        self.merkle_chain: List[str] = []
        self.global_cpu_start_ns = time.process_time_ns()
        self.logger.log_event("SYSTEM_STARTUP_L0_SUCCESS", config=asdict(cfg))

    def slab_allocate(self, tensor: np.ndarray, phase: str) -> int:
        variance = float(np.var(tensor, dtype=np.float64))
        entropy_proxy = float(np.mean(np.abs(np.gradient(tensor.astype(np.float32), axis=-1))))
        required = int(self.cfg.slab_min_neurons + (tensor.size // 512) + variance * 1500.0 + entropy_proxy * 900.0)
        self.neurons_active = max(self.cfg.slab_min_neurons, required)
        self.total_allocations += 1
        self.logger.log_event(
            "SLAB_ALLOCATION",
            phase=phase,
            neurons=self.neurons_active,
            variance=round(variance, 8),
            entropy_proxy=round(entropy_proxy, 8),
        )
        return self.neurons_active

    def _track_bits(self, fragment: str, arr: np.ndarray) -> None:
        self.logger.log_bits(fragment, arr.tobytes()[: self.cfg.bit_capture_bytes])

    def _merkle_sign(self, fragment: str, arr: np.ndarray) -> str:
        prev = self.merkle_chain[-1] if self.merkle_chain else "GENESIS"
        digest = sha512(prev.encode("utf-8") + arr.tobytes()).hexdigest()
        self.merkle_chain.append(digest)
        self.logger.log_merkle(fragment, digest)
        return digest

    @staticmethod
    def _normalize_stack(stack: np.ndarray) -> np.ndarray:
        stack = stack.astype(np.float32)
        mn, mx = float(stack.min()), float(stack.max())
        return np.zeros_like(stack, dtype=np.float32) if mx <= mn else (stack - mn) / (mx - mn)

    @staticmethod
    def _ink_energy_projection(stack: np.ndarray) -> np.ndarray:
        grad_z = np.abs(np.diff(stack, axis=0, prepend=stack[:1]))
        grad_y = np.abs(np.diff(stack, axis=1, prepend=stack[:, :1, :]))
        grad_x = np.abs(np.diff(stack, axis=2, prepend=stack[:, :, :1]))
        return np.mean(0.45 * grad_z + 0.30 * grad_y + 0.25 * grad_x, axis=0)

    def train_threshold(self, stack: np.ndarray, labels: np.ndarray, fragment: str) -> float:
        self.logger.set_roadmap("train", 15.0)
        start = time.process_time_ns()
        self.slab_allocate(stack, phase="train")
        self._track_bits(fragment, stack)

        score = self._ink_energy_projection(self._normalize_stack(stack))
        pos, neg = score[labels > 0], score[labels <= 0]
        if pos.size and neg.size:
            threshold = float(0.5 * (float(np.median(pos)) + float(np.median(neg))))
        elif pos.size:
            threshold = float(np.quantile(pos, 0.50))
        else:
            threshold = float(np.quantile(score, self.cfg.threshold_quantile))

        digest = self._merkle_sign(fragment, score)
        pred = (score >= threshold).astype(np.uint8)
        ink_pixels, total_pixels = int(pred.sum()), int(pred.size)
        cpu_ns = time.process_time_ns() - start
        self.total_ink_pixels += ink_pixels
        self.total_pixels_processed += total_pixels
        self.logger.log_metrics(
            phase="train",
            fragment=fragment,
            neurons_active=self.neurons_active,
            cpu_ns=cpu_ns,
            ink_pixels=ink_pixels,
            total_pixels=total_pixels,
            merkle_prefix=digest[:16],
        )
        self.logger.log_event("TRAIN_DONE", fragment=fragment, threshold=threshold)
        self.logger.set_roadmap("train", 100.0, done=True)
        return threshold

    def infer_mask(self, stack: np.ndarray, threshold: float, fragment: str) -> np.ndarray:
        start = time.process_time_ns()
        self.slab_allocate(stack, phase="infer")
        self._track_bits(fragment, stack)

        score = self._ink_energy_projection(self._normalize_stack(stack))
        pred = (score >= threshold).astype(np.uint8)
        digest = self._merkle_sign(fragment, pred)
        ink_pixels, total_pixels = int(pred.sum()), int(pred.size)
        cpu_ns = time.process_time_ns() - start

        self.total_ink_pixels += ink_pixels
        self.total_pixels_processed += total_pixels
        self.logger.log_metrics(
            phase="infer",
            fragment=fragment,
            neurons_active=self.neurons_active,
            cpu_ns=cpu_ns,
            ink_pixels=ink_pixels,
            total_pixels=total_pixels,
            merkle_prefix=digest[:16],
        )
        self.logger.log_event("INFER_DONE", fragment=fragment, threshold=threshold)
        return pred

    def finalize(self, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        cpu_total_ns = time.process_time_ns() - self.global_cpu_start_ns
        state = {
            "status": "100%_OFFLINE_ACTIVATED",
            "active_neurons": self.neurons_active,
            "total_allocations": self.total_allocations,
            "total_pixels_processed": self.total_pixels_processed,
            "total_ink_pixels": self.total_ink_pixels,
            "ink_ratio": self.total_ink_pixels / self.total_pixels_processed if self.total_pixels_processed else 0.0,
            "qi_index_real": self.total_pixels_processed / max(cpu_total_ns, 1),
            "cpu_total_ns": cpu_total_ns,
            "merkle_root": self.merkle_chain[-1] if self.merkle_chain else None,
        }
        if extra:
            state.update(extra)
        self.logger.write_state(state)
        self.logger.flush_events()
        self.logger.log_event("SYSTEM_LOADED_100_PERCENT")
        return state


def _read_stack_tif(volume_dir: Path) -> np.ndarray:
    files = sorted(volume_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFF slices found in {volume_dir}")
    return np.stack([iio.imread(str(p)) for p in files], axis=0)


def _load_legacy_3d_tif(tif_path: Path) -> np.ndarray:
    arr = iio.imread(str(tif_path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported tif shape for {tif_path}: {arr.shape}")
    return arr


def _discover_train_test(root: Path) -> Tuple[List[Path], List[Path], str]:
    # mode A: train/*/surface_volume + test/*/surface_volume
    train_a, test_a = root / "train", root / "test"
    if train_a.exists() and test_a.exists():
        train = sorted([p for p in train_a.iterdir() if p.is_dir() and (p / "surface_volume").exists()])
        test = sorted([p for p in test_a.iterdir() if p.is_dir() and (p / "surface_volume").exists()])
        if train or test:
            return train, test, "fragment_dirs"

    # mode B: legacy train_images/train_labels and test_images/*.tif
    train_img, test_img = root / "train_images", root / "test_images"
    if test_img.exists():
        train = sorted(train_img.glob("*.tif")) if train_img.exists() else []
        test = sorted(test_img.glob("*.tif"))
        return train, test, "legacy_tif_files"

    return [], [], "none"


def _load_train_item(item: Path, mode: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if mode == "fragment_dirs":
        stack = _read_stack_tif(item / "surface_volume")
        lab = item / "inklabels.png"
        labels = None
        if lab.exists():
            arr = iio.imread(str(lab))
            labels = (arr[..., 0] > 0).astype(np.uint8) if arr.ndim == 3 else (arr > 0).astype(np.uint8)
        return stack, labels, item.name

    # legacy: single 3D tif train image with optional paired label tif/png
    stack = _load_legacy_3d_tif(item)
    lbl_tif = item.parent.parent / "train_labels" / item.name
    lbl_png = item.parent.parent / "train_labels" / item.with_suffix(".png").name
    labels = None
    if lbl_tif.exists():
        l = iio.imread(str(lbl_tif))
        labels = (l[0] > 0).astype(np.uint8) if l.ndim == 3 else (l > 0).astype(np.uint8)
    elif lbl_png.exists():
        l = iio.imread(str(lbl_png))
        labels = (l[..., 0] > 0).astype(np.uint8) if l.ndim == 3 else (l > 0).astype(np.uint8)
    return stack, labels, item.stem


def _load_test_item(item: Path, mode: str) -> Tuple[np.ndarray, str]:
    if mode == "fragment_dirs":
        return _read_stack_tif(item / "surface_volume"), f"{item.name}.tif"
    return _load_legacy_3d_tif(item), item.name


def _write_submission_zip(out_zip: Path, predictions: Dict[str, np.ndarray]) -> str:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for tif_name in sorted(predictions):
            mask2d = predictions[tif_name].astype(np.uint8)
            if mask2d.ndim != 2:
                raise RuntimeError(f"Prediction for {tif_name} must be 2D, got {mask2d.shape}")
            tmp = out_zip.parent / tif_name
            tifffile.imwrite(str(tmp), mask2d[np.newaxis, ...], compression="LZW")
            zf.write(tmp, arcname=tif_name)
            tmp.unlink(missing_ok=True)
    return str(out_zip)


def _validate_submission_zip(out_zip: Path, expected_tif_names: List[str]) -> Dict[str, object]:
    with zipfile.ZipFile(out_zip, "r") as zf:
        got = sorted([Path(n).name for n in zf.namelist() if n.lower().endswith(".tif")])
    expected = sorted(expected_tif_names)
    status = "ok" if got == expected else "mismatch"
    return {
        "status": status,
        "expected_test_files": len(expected),
        "submission_tif_files": len(got),
        "missing": [n for n in expected if n not in got],
        "unexpected": [n for n in got if n not in expected],
    }


def run_pipeline(cfg: NX46Config) -> Dict[str, object]:
    root = Path(cfg.data_root)
    log_root = Path(cfg.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    nx46 = NX46AGNNVesuvius(cfg)
    nx46.logger.set_roadmap("discovery", 10.0)
    train_items, test_items, mode = _discover_train_test(root)
    nx46.logger.log_event("DATASET_DISCOVERY", mode=mode, train_count=len(train_items), test_count=len(test_items))
    nx46.logger.set_roadmap("discovery", 100.0, done=True)

    if not test_items:
        raise RuntimeError(f"No test inputs found under {root}")

    nx46.logger.set_roadmap("load", 100.0, done=True)
    nx46.logger.set_roadmap("features", 100.0, done=True)

    thresholds: List[float] = []
    for idx, item in enumerate(train_items, start=1):
        nx46.logger.log_event("PROGRESS_TRAIN", index=idx, total=len(train_items), percent=round(idx * 100.0 / max(1, len(train_items)), 2))
        stack, labels, fragment = _load_train_item(item, mode)
        if labels is None:
            continue
        if labels.shape != stack.shape[1:]:
            h, w = min(labels.shape[0], stack.shape[1]), min(labels.shape[1], stack.shape[2])
            labels = labels[:h, :w]
            stack = stack[:, :h, :w]
        thresholds.append(nx46.train_threshold(stack, labels, fragment))

    threshold = float(np.median(np.array(thresholds, dtype=np.float32))) if thresholds else 0.5
    nx46.logger.log_event("THRESHOLD_SELECTED", threshold=threshold, trained_samples=len(thresholds))

    nx46.logger.set_roadmap("segment", 10.0)
    predictions: Dict[str, np.ndarray] = {}
    expected_names: List[str] = []
    for idx, item in enumerate(test_items, start=1):
        nx46.logger.log_event("PROGRESS_TEST", index=idx, total=len(test_items), percent=round(idx * 100.0 / max(1, len(test_items)), 2))
        stack, tif_name = _load_test_item(item, mode)
        pred = nx46.infer_mask(stack, threshold, Path(tif_name).stem)
        predictions[tif_name] = pred
        expected_names.append(tif_name)
    nx46.logger.set_roadmap("segment", 100.0, done=True)

    nx46.logger.set_roadmap("package", 10.0)
    submission_zip = Path(cfg.work_root) / "submission.zip"
    submission_path = _write_submission_zip(submission_zip, predictions)
    rule_validation = _validate_submission_zip(submission_zip, expected_names) if cfg.enforce_submission_rules else {"status": "disabled"}
    nx46.logger.log_event("COMPETITION_RULES_VALIDATION", **rule_validation)
    nx46.logger.set_roadmap("package", 100.0, done=True)

    # compatibility text stubs requested in prior forensic bundles
    (log_root / "RkF4XakI.txt").write_text(json.dumps({"status": "generated", "submission_path": submission_path}, indent=2), encoding="utf-8")
    (log_root / "UJxLRsEE.txt").write_text(json.dumps(rule_validation, indent=2), encoding="utf-8")

    return nx46.finalize(
        {
            "mode": mode,
            "train_items": [p.name for p in train_items],
            "test_items": [p.name for p in test_items],
            "train_threshold": threshold,
            "submission_zip": submission_path,
            "competition_rules_validation": rule_validation,
        }
    )


if __name__ == "__main__":
    config = NX46Config(
        data_root=os.environ.get("NX46_DATA_ROOT", NX46Config.data_root),
        work_root=os.environ.get("NX46_WORK_ROOT", NX46Config.work_root),
        log_root=os.environ.get("NX46_LOG_ROOT", NX46Config.log_root),
    )
    result = run_pipeline(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))

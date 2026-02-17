"""NX-46 Vesuvius Kaggle-ready v5 (offline-first, forensic, regression-safe).

V5 objectives:
- Keep all validated v3/v4 compatibility outputs (submission.zip + forensic logs + compatibility txt files).
- Fix roadmap status inconsistency (`finalize_forensics` saved at 60% in state.json in v4 logs).
- Reduce avoidable runtime by skipping heavy train stack loading when labels are absent.
- Add explicit training strategy telemetry (supervised vs fallback quantile).
- Preserve strict Kaggle compliance and offline dependency bootstrap.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from hashlib import sha512
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def install_offline(package_name: str) -> None:
    exact_wheel_dir = Path("/kaggle/input/datasets/ndarray2000/nx47-dependencies")
    fallback_wheel_dir = Path("/kaggle/input/nx47-dependencies")

    exact_wheels = {
        "imagecodecs": exact_wheel_dir / "imagecodecs-2026.1.14-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "numpy": exact_wheel_dir / "numpy-2.4.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "tifffile": exact_wheel_dir / "tifffile-2026.1.28-py3-none-any.whl",
    }

    if package_name == "numpy" and importlib.util.find_spec("numpy") is not None:
        return

    if package_name in exact_wheels and exact_wheels[package_name].exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", str(exact_wheels[package_name])])
            return
        except subprocess.CalledProcessError:
            pass

    for wheel_dir in (exact_wheel_dir, fallback_wheel_dir):
        if wheel_dir.exists():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={wheel_dir}", package_name]
            )
            return

    raise RuntimeError(
        f"Offline dependency directory not found for {package_name}. "
        f"Checked: {exact_wheel_dir} and {fallback_wheel_dir}"
    )


def bootstrap_dependencies_fail_fast() -> None:
    install_offline("numpy")
    install_offline("imagecodecs")
    install_offline("tifffile")


bootstrap_dependencies_fail_fast()

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import imageio.v3 as iio  # noqa: E402


def ensure_imagecodecs() -> bool:
    if importlib.util.find_spec("imagecodecs") is not None:
        return True

    try:
        install_offline("imagecodecs")
    except Exception:
        return False

    if importlib.util.find_spec("imagecodecs") is None:
        return False

    global tifffile
    tifffile = importlib.reload(tifffile)
    return True


def read_tiff_lzw_safe(path: Path) -> np.ndarray:
    try:
        return np.asarray(tifffile.imread(path))
    except ValueError as exc:
        if "requires the 'imagecodecs' package" not in str(exc):
            raise

    ensure_imagecodecs()
    try:
        return np.asarray(tifffile.imread(path))
    except ValueError as exc:
        if "requires the 'imagecodecs' package" not in str(exc):
            raise

    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("LZW TIFF read failed and Pillow fallback unavailable")

    from PIL import Image, ImageSequence

    with Image.open(path) as img:
        frames = [np.array(frame, dtype=np.float32) for frame in ImageSequence.Iterator(img)]
    if not frames:
        raise RuntimeError(f"No frames decoded from TIFF: {path}")
    return np.stack(frames, axis=0)


@dataclass
class NX46Config:
    data_root: str = "/kaggle/input/vesuvius-challenge-ink-detection"
    work_root: str = "/kaggle/working/nx46_vesuvius"
    seed: int = 46
    bit_capture_bytes: int = 256
    threshold_quantile: float = 0.985
    slab_min_neurons: int = 128
    auto_discover_data_root: bool = True
    strict_competition_mode: bool = True
    max_train_items: int = 512


class ProgressRoadmap:
    def __init__(self) -> None:
        self.steps: Dict[str, float] = {
            "audit_discovery": 0.0,
            "train_thresholds": 0.0,
            "infer_predictions": 0.0,
            "package_submission": 0.0,
            "finalize_forensics": 0.0,
        }

    def update(self, step: str, pct: float) -> None:
        self.steps[step] = max(0.0, min(100.0, float(pct)))
        print(f"[ROADMAP] {step}: {self.steps[step]:.1f}%")

    def as_dict(self) -> Dict[str, float]:
        return dict(self.steps)


class HFBL360Logger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

        self.forensic_log = root / "forensic_ultra.log"
        self.core_log = root / "nx-46-vesuvius-core.log"
        self.kaggle_log = root / "nx46-vesuvius-core-kaggle-ready.log"
        self.metrics_csv = root / "metrics.csv"
        self.state_json = root / "state.json"
        self.bit_log = root / "bit_capture.log"
        self.merkle_log = root / "merkle_chain.log"
        self.discovery_json = root / "dataset_discovery_inventory.json"

        with self.metrics_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
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

    def _append(self, line: str) -> None:
        for p in (self.forensic_log, self.core_log, self.kaggle_log):
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_event(self, event: str, **data: object) -> None:
        payload = json.dumps(data, ensure_ascii=False) if data else ""
        self._append(f"{time.time_ns()} | {event}{(' | ' + payload) if payload else ''}")

    def log_bits(self, fragment: str, payload: bytes) -> None:
        bits = "".join(f"{b:08b}" for b in payload)
        with self.bit_log.open("a", encoding="utf-8") as f:
            f.write(f"{time.time_ns()} | {fragment} | {bits}\n")

    def log_merkle(self, fragment: str, digest: str) -> None:
        with self.merkle_log.open("a", encoding="utf-8") as f:
            f.write(f"{time.time_ns()} | {fragment} | {digest}\n")

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

    def write_state(self, state: Dict[str, object]) -> None:
        state = dict(state)
        state["timestamp_ns"] = time.time_ns()
        self.state_json.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_discovery_inventory(self, payload: Dict[str, object]) -> None:
        self.discovery_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class NX46AGNNVesuvius:
    def __init__(self, cfg: NX46Config, logs_root: Path) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.logger = HFBL360Logger(logs_root)
        self.neurons_active = 0
        self.total_allocations = 0
        self.total_pixels_processed = 0
        self.total_ink_pixels = 0
        self.merkle_chain: List[str] = []
        self.global_cpu_start_ns = time.process_time_ns()
        self.logger.log_event("SYSTEM_STARTUP_L0_SUCCESS", config=cfg.__dict__)

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
        x = stack.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    @staticmethod
    def _ink_energy_projection(stack: np.ndarray) -> np.ndarray:
        grad_z = np.abs(np.diff(stack, axis=0, prepend=stack[:1]))
        grad_y = np.abs(np.diff(stack, axis=1, prepend=stack[:, :1, :]))
        grad_x = np.abs(np.diff(stack, axis=2, prepend=stack[:, :, :1]))
        return np.mean(0.45 * grad_z + 0.30 * grad_y + 0.25 * grad_x, axis=0)

    def score_projection(self, stack: np.ndarray) -> np.ndarray:
        return self._ink_energy_projection(self._normalize_stack(stack))

    def train_threshold(self, stack: np.ndarray, labels: np.ndarray, fragment: str) -> float:
        start = time.process_time_ns()
        self.slab_allocate(stack, phase="train")
        self._track_bits(fragment, stack)

        score = self.score_projection(stack)
        pos = score[labels > 0]
        neg = score[labels <= 0]
        if pos.size and neg.size:
            threshold = float(0.5 * (float(np.median(pos)) + float(np.median(neg))))
        elif pos.size:
            threshold = float(np.quantile(pos, 0.50))
        else:
            threshold = float(np.quantile(score, self.cfg.threshold_quantile))

        pred = (score >= threshold).astype(np.uint8)
        digest = self._merkle_sign(fragment, score)
        cpu_ns = time.process_time_ns() - start

        ink_pixels = int(pred.sum())
        total_pixels = int(pred.size)
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
        return threshold

    def infer_mask(self, stack: np.ndarray, threshold: float, fragment: str) -> np.ndarray:
        start = time.process_time_ns()
        self.slab_allocate(stack, phase="infer")
        self._track_bits(fragment, stack)

        score = self.score_projection(stack)
        pred = (score >= threshold).astype(np.uint8)
        digest = self._merkle_sign(fragment, pred)
        cpu_ns = time.process_time_ns() - start

        ink_pixels = int(pred.sum())
        total_pixels = int(pred.size)
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
        self.logger.log_event("INFER_DONE", fragment=fragment)
        return pred

    def finalize(self, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        cpu_total_ns = time.process_time_ns() - self.global_cpu_start_ns
        state = {
            "runtime_status": "offline_activated",
            "pipeline_status": "success",
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
        self.logger.log_event("SYSTEM_LOADED_100_PERCENT")
        return state


def _to_stack(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        return x[np.newaxis, ...]
    if x.ndim == 3:
        return x
    raise RuntimeError(f"Unsupported TIFF shape: {x.shape}")


def _read_fragment_stack(fragment_dir: Path) -> np.ndarray:
    volume_dir = fragment_dir / "surface_volume"
    files = sorted(volume_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFF slices found in {volume_dir}")
    return np.stack([_to_stack(read_tiff_lzw_safe(p))[0] for p in files], axis=0)


def _load_label(fragment_dir: Path) -> Optional[np.ndarray]:
    png = fragment_dir / "inklabels.png"
    if not png.exists() or iio is None:
        return None
    arr = np.asarray(iio.imread(str(png)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def _read_sample_submission_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        id_col = "Id" if "Id" in cols else ("id" if "id" in cols else None)
        if id_col is None:
            return ids
        for row in reader:
            v = str(row[id_col]).strip()
            if v:
                ids.append(v)
    return ids


def _dataset_inventory(root: Path) -> Dict[str, object]:
    if not root.exists():
        return {"root": str(root), "exists": False}

    all_files = [p for p in root.rglob("*") if p.is_file()]
    suffix_stats: Dict[str, int] = {}
    folders = set()
    for p in all_files:
        suffix = p.suffix.lower() or "<noext>"
        suffix_stats[suffix] = suffix_stats.get(suffix, 0) + 1
        folders.add(str(p.parent.relative_to(root)))

    return {
        "root": str(root),
        "exists": True,
        "total_assets": len(all_files),
        "folders": sorted(folders),
        "suffix_stats": suffix_stats,
    }


def _discover_layout(root: Path) -> Tuple[str, List[Path], List[Path]]:
    test_images = root / "test_images"
    if test_images.exists():
        train_images = root / "train_images"
        train = sorted(train_images.rglob("*.tif")) if train_images.exists() else []
        test = sorted(test_images.rglob("*.tif"))
        if test:
            return "competition_test_images", train, test

    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.exists() or test_dir.exists():
        train = sorted([p for p in train_dir.iterdir() if p.is_dir() and (p / "surface_volume").exists()]) if train_dir.exists() else []
        test = sorted([p for p in test_dir.iterdir() if p.is_dir() and (p / "surface_volume").exists()]) if test_dir.exists() else []
        if test:
            return "fragment_dirs", train, test

    train_legacy = sorted((root / "train_images").glob("*.tif")) if (root / "train_images").exists() else []
    test_legacy = sorted((root / "test_images").glob("*.tif")) if (root / "test_images").exists() else []
    if test_legacy:
        return "legacy_tif_files", train_legacy, test_legacy

    return "empty", [], []


def _candidate_roots(primary_root: Path) -> List[Path]:
    candidates: List[Path] = []

    def add(p: Path) -> None:
        if p.exists() and p not in candidates:
            candidates.append(p)

    add(primary_root)
    add(Path("/kaggle/input/vesuvius-challenge-ink-detection"))
    add(Path("/kaggle/input/vesuvius-challenge-surface-detection"))
    add(Path("/kaggle/input/competitions/vesuvius-challenge-surface-detection"))

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in sorted(kaggle_input.iterdir()):
            if p.is_dir() and "vesuvius" in p.name.lower():
                add(p)

    competitions = Path("/kaggle/input/competitions")
    if competitions.exists():
        for p in sorted(competitions.iterdir()):
            if p.is_dir() and "vesuvius" in p.name.lower():
                add(p)

    return candidates


def _resolve_root(primary_root: Path, auto_discover: bool) -> Tuple[Path, str, List[Dict[str, object]]]:
    roots = _candidate_roots(primary_root) if auto_discover else [primary_root]
    attempts: List[Dict[str, object]] = []

    for root in roots:
        mode, train, test = _discover_layout(root)
        attempts.append({"root": str(root), "mode": mode, "train": len(train), "test": len(test)})
        if test:
            return root, mode, attempts

    return primary_root, "empty", attempts


def _label_candidates_for_train_tif(item: Path) -> List[Path]:
    root = item.parent.parent
    return [
        root / "train_labels" / item.name,
        root / "train_labels" / item.with_suffix(".png").name,
    ]


def _quick_has_label(item: Path, mode: str) -> bool:
    if mode == "fragment_dirs":
        return (item / "inklabels.png").exists()
    return any(c.exists() for c in _label_candidates_for_train_tif(item))


def _load_train_item(item: Path, mode: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    if mode == "fragment_dirs":
        stack = _read_fragment_stack(item)
        labels = _load_label(item)
        return stack, labels, item.name

    stack = _to_stack(read_tiff_lzw_safe(item))

    labels = None
    labels_tif, labels_png = _label_candidates_for_train_tif(item)
    if labels_tif.exists():
        l = read_tiff_lzw_safe(labels_tif)
        labels = (l[0] > 0).astype(np.uint8) if l.ndim == 3 else (l > 0).astype(np.uint8)
    elif labels_png.exists() and iio is not None:
        l = np.asarray(iio.imread(str(labels_png)))
        labels = (l[..., 0] > 0).astype(np.uint8) if l.ndim == 3 else (l > 0).astype(np.uint8)

    return stack, labels, item.stem


def _load_test_item(item: Path, mode: str) -> Tuple[np.ndarray, str]:
    if mode == "fragment_dirs":
        return _read_fragment_stack(item), f"{item.name}.tif"
    return _to_stack(read_tiff_lzw_safe(item)), item.name


def _write_submission_csv(out_csv: Path, sample_csv: Path, predictions: Dict[str, np.ndarray]) -> Optional[str]:
    ids = _read_sample_submission_ids(sample_csv)
    if not ids:
        return None

    flat = np.concatenate([predictions[k].reshape(-1).astype(np.uint8) for k in sorted(predictions)])
    n = min(len(ids), len(flat))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Predicted"])
        for i in range(n):
            w.writerow([ids[i], int(flat[i])])
    return str(out_csv)


def _write_submission_zip(out_zip: Path, predictions: Dict[str, np.ndarray]) -> str:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_zip.parent / "submission_masks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    names: List[str] = []
    for tif_name, mask in predictions.items():
        # Kaggle-compatible binary mask encoded on full uint8 dynamic range (0 or 255).
        mask2d = ((mask > 0).astype(np.uint8) * 255).astype(np.uint8)
        p = tmp_dir / tif_name
        tifffile.imwrite(str(p), mask2d[np.newaxis, ...], compression="LZW")
        names.append(tif_name)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(names):
            zf.write(tmp_dir / name, arcname=name)

    return str(out_zip)


def _validate_zip(out_zip: Path, expected_tif_names: Sequence[str]) -> Dict[str, object]:
    with zipfile.ZipFile(out_zip, "r") as zf:
        got = sorted([Path(n).name for n in zf.namelist() if n.lower().endswith(".tif")])
    exp = sorted(expected_tif_names)
    return {
        "status": "ok" if got == exp else "mismatch",
        "expected_test_files": len(exp),
        "submission_tif_files": len(got),
        "missing": sorted(set(exp) - set(got)),
        "unexpected": sorted(set(got) - set(exp)),
    }


def _publish_submission_aliases(primary_zip: Path, work_root: Path) -> List[str]:
    aliases = [
        Path("/kaggle/working/submission.zip"),
        Path("/kaggle/working/nx46_vesuvius/submission.zip"),
        Path("submission.zip"),
        Path("nx46_vesuvius/submission.zip"),
        work_root / "submission.zip",
    ]

    published: List[str] = []
    for alias in aliases:
        if alias.resolve() == primary_zip.resolve():
            published.append(str(alias))
            continue
        alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary_zip, alias)
        published.append(str(alias))
    return sorted(set(published))


def _calibrate_threshold_from_unlabeled(
    model: NX46AGNNVesuvius,
    test_items: Sequence[Path],
    mode: str,
    q: float,
) -> float:
    sampled = list(test_items[: min(2, len(test_items))])
    if not sampled:
        return 0.5

    q = min(0.999, max(0.50, q))
    per_item: List[float] = []
    for item in sampled:
        stack, tif_name = _load_test_item(item, mode)
        model.slab_allocate(stack, phase="train_fallback_probe")
        model._track_bits(Path(tif_name).stem, stack)
        score = model.score_projection(stack)
        digest = model._merkle_sign(f"probe_{Path(tif_name).stem}", score)
        model.logger.log_metrics(
            phase="train_fallback_probe",
            fragment=Path(tif_name).stem,
            neurons_active=model.neurons_active,
            cpu_ns=0,
            ink_pixels=int((score >= np.quantile(score, q)).sum()),
            total_pixels=int(score.size),
            merkle_prefix=digest[:16],
        )
        per_item.append(float(np.quantile(score, q)))

    return float(np.median(np.asarray(per_item, dtype=np.float32)))


def run_pipeline(cfg: NX46Config) -> Dict[str, object]:
    configured_root = Path(cfg.data_root)
    work = Path(cfg.work_root)
    logs_root = work / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    roadmap = ProgressRoadmap()
    nx46 = NX46AGNNVesuvius(cfg, logs_root)

    effective_root, mode, attempts = _resolve_root(configured_root, cfg.auto_discover_data_root)
    mode, train_items, test_items = _discover_layout(effective_root)

    inventory = _dataset_inventory(effective_root)
    nx46.logger.write_discovery_inventory({"attempts": attempts, "inventory": inventory})
    nx46.logger.log_event("DATASET_DISCOVERY", mode=mode, configured_root=str(configured_root), effective_root=str(effective_root), attempts=attempts)
    roadmap.update("audit_discovery", 100.0)

    if not test_items:
        nx46.logger.log_event("NO_TEST_INPUTS_FOUND", attempts=attempts)
        raise RuntimeError(
            "No test inputs found. Checked roots: "
            + ", ".join([a["root"] for a in attempts])
            + ". Configure NX46_DATA_ROOT to the dataset root containing test_images or test/<fragment>."
        )

    usable_train_items = [p for p in train_items if _quick_has_label(p, mode)]
    skipped_no_label = max(0, len(train_items) - len(usable_train_items))
    if cfg.max_train_items > 0:
        usable_train_items = usable_train_items[: cfg.max_train_items]

    thresholds: List[float] = []
    total_train = max(1, len(usable_train_items))
    for idx, item in enumerate(usable_train_items, start=1):
        roadmap.update("train_thresholds", idx * 100.0 / total_train)
        nx46.logger.log_event("PROGRESS_TRAIN", index=idx, total=len(usable_train_items))
        stack, labels, name = _load_train_item(item, mode)
        if labels is None:
            continue
        if labels.shape != stack.shape[1:]:
            h = min(labels.shape[0], stack.shape[1])
            w = min(labels.shape[1], stack.shape[2])
            labels = labels[:h, :w]
            stack = stack[:, :h, :w]
        thresholds.append(nx46.train_threshold(stack, labels, name))

    training_strategy = "supervised"
    if thresholds:
        threshold = float(np.median(np.asarray(thresholds, dtype=np.float32)))
    else:
        threshold = _calibrate_threshold_from_unlabeled(nx46, test_items, mode, cfg.threshold_quantile)
        training_strategy = "fallback_quantile_probe"

    nx46.logger.log_event(
        "THRESHOLD_SELECTED",
        threshold=threshold,
        trained_samples=len(thresholds),
        training_strategy=training_strategy,
        skipped_no_label=skipped_no_label,
        max_train_items=cfg.max_train_items,
    )

    predictions: Dict[str, np.ndarray] = {}
    expected_names: List[str] = []

    total_test = max(1, len(test_items))
    for idx, item in enumerate(test_items, start=1):
        roadmap.update("infer_predictions", idx * 100.0 / total_test)
        nx46.logger.log_event("PROGRESS_TEST", index=idx, total=len(test_items))
        stack, tif_name = _load_test_item(item, mode)
        pred = nx46.infer_mask(stack, threshold, Path(tif_name).stem)
        predictions[tif_name] = pred
        expected_names.append(tif_name)

    roadmap.update("package_submission", 20.0)

    sample_csv_candidates = [
        effective_root / "sample_submission.csv",
        Path("/kaggle/input/vesuvius-challenge-ink-detection/sample_submission.csv"),
        Path("/kaggle/input/vesuvius-challenge-surface-detection/sample_submission.csv"),
        Path("/kaggle/input/competitions/vesuvius-challenge-surface-detection/sample_submission.csv"),
    ]
    sample_csv = next((p for p in sample_csv_candidates if p.exists()), None)

    submission_csv: Optional[str] = None
    if sample_csv is not None:
        submission_csv = _write_submission_csv(work / "submission.csv", sample_csv, predictions)

    submission_zip = _write_submission_zip(work / "submission.zip", predictions)
    submission_zip_aliases = _publish_submission_aliases(Path(submission_zip), work)
    validation = _validate_zip(Path(submission_zip), expected_names)
    nx46.logger.log_event("COMPETITION_RULES_VALIDATION", **validation)
    nx46.logger.log_event("SUBMISSION_PATHS_PUBLISHED", primary=submission_zip, aliases=submission_zip_aliases)
    roadmap.update("package_submission", 100.0)

    if cfg.strict_competition_mode and validation["status"] != "ok":
        raise RuntimeError(f"Submission zip validation failed: {validation}")

    roadmap.update("finalize_forensics", 100.0)
    result = nx46.finalize(
        {
            "status": "100%_OFFLINE_ACTIVATED",
            "layout_detected": mode,
            "configured_data_root": str(configured_root),
            "effective_data_root": str(effective_root),
            "discovery_attempts": attempts,
            "train_items_total": len(train_items),
            "train_items_with_labels": len(usable_train_items),
            "train_items_skipped_no_label": skipped_no_label,
            "train_items": [p.name for p in train_items],
            "test_items": [p.name for p in test_items],
            "train_threshold": threshold,
            "training_strategy": training_strategy,
            "submission_csv": submission_csv,
            "submission_zip": submission_zip,
            "submission_zip_aliases": submission_zip_aliases,
            "zip_members_validated": validation["status"] == "ok",
            "zip_missing": validation["missing"],
            "zip_extra": validation["unexpected"],
            "competition_rules_validation": validation,
            "roadmap_percent": roadmap.as_dict(),
        }
    )

    (logs_root / "RkF4XakI.txt").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (logs_root / "UJxLRsEE.txt").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    nx46.logger.log_event("EXEC_COMPLETE", submission=submission_zip)
    return result


if __name__ == "__main__":
    config = NX46Config(
        data_root=os.environ.get("NX46_DATA_ROOT", NX46Config.data_root),
        work_root=os.environ.get("NX46_WORK_ROOT", NX46Config.work_root),
        max_train_items=int(os.environ.get("NX46_MAX_TRAIN_ITEMS", str(NX46Config.max_train_items))),
    )
    out = run_pipeline(config)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"READY: {out['submission_zip']}")

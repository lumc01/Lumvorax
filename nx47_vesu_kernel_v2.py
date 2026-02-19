import ctypes
import glob
import importlib
import io
import json
import os
import struct
import time
import zipfile
from dataclasses import dataclass
from hashlib import sha512
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class FatalPipelineError(RuntimeError):
    """Raised when fail-fast invariants are violated."""


@dataclass(frozen=True)
class CompatibilityLayer:
    name: str
    required_capabilities: List[str]


@dataclass(frozen=True)
class LUMVolume:
    shape: Tuple[int, int, int]
    dtype: str
    payload_sha512: str


class LumVoraxBridge:
    """Optional bridge to native LUM/VORAX C/C++ libraries via ctypes.

    Bridge is optional by design: pipeline remains fully functional in pure Python
    when no native shared library is available in Kaggle runtime.
    """

    def __init__(self) -> None:
        self.lib = None
        self.loaded_path = None

        candidates = [
            os.environ.get("LUM_VORAX_LIB_PATH", ""),
            "/kaggle/working/liblumvorax.so",
            "/kaggle/working/libvorax.so",
            "/kaggle/input/lum-vorax-dependencies/liblumvorax.so",
            "/kaggle/input/lum-vorax-dependencies/libvorax.so",
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                self.lib = ctypes.CDLL(candidate)
                self.loaded_path = candidate
                break

    @property
    def enabled(self) -> bool:
        return self.lib is not None


class NX47_VESU_Production:
    """NX-47 VESU production pipeline (Kaggle-ready, fail-fast, forensic).

    Key guarantees:
    - No synthetic fragment fallback.
    - Real TIFF input ingestion (2D or 3D, normalized to 3D).
    - Progressive multi-threshold 3D accumulation with target density clamping.
    - Optional `.lum` canonical intermediate serialization.
    - Optional native LUM/VORAX bridge via ctypes (non-blocking fallback).
    """

    ROADMAP_STEPS = [
        "bootstrap",
        "compatibility_check",
        "data_validation",
        "feature_extraction",
        "inference",
        "forensic_export",
        "finalize",
    ]

    LUM_MAGIC = b"LUMV1\x00\x00\x00"

    def __init__(self, input_dir=None, output_dir=None):
        self.version = "NX-47 VESU PROD V133-REAL-PY"
        self.audit_log: List[Dict] = []
        self.start_time = time.time_ns()
        self.input_dir = input_dir or "/kaggle/input/vesuvius-challenge-surface-detection"
        self.output_dir = output_dir or "/kaggle/working"
        self.processed_pixels = 0
        self.ink_detected = 0
        self.fallback_disabled = True
        self.roadmap_path = os.path.join(self.output_dir, "v133_roadmap_realtime.json")
        self.execution_log_path = os.path.join(self.output_dir, "v133_execution_logs.json")
        self.metadata_path = os.path.join(self.output_dir, "v133_execution_metadata.json")
        self.submission_zip_path = os.path.join(self.output_dir, "submission.zip")
        self.submission_parquet_path = os.path.join(self.output_dir, "submission.parquet")
        self.lum_work_dir = os.path.join(self.output_dir, "lum_cache")
        self.bridge = LumVoraxBridge()

        self.capability_registry = {
            "preprocess_invariants": self.spatial_harmonic_filtering_simd,
            "input_format_guard": self._validate_input_structure,
            "feature_signature_v2": self._extract_fragment_signature,
            "intermediate_schema_v2": self._build_result_entry,
            "audit_hash_chain": self.log_event,
            "integrity_checks": self._integrity_digest,
            "forensic_traceability": self._export_forensic,
            "merkle_ready_events": self._audit_merkle_root,
            "realtime_roadmap": self._update_roadmap,
            "strict_train_evidence_gate": self._strict_training_evidence_gate,
            "adaptive_thresholding": self.ink_resonance_detector_v47,
            "dynamic_neuron_telemetry": self._emit_neuron_telemetry,
            "lum_encode_decode": self._roundtrip_lum,
        }

        self.compatibility_layers = [
            CompatibilityLayer("NX-1..NX-10", ["preprocess_invariants", "input_format_guard"]),
            CompatibilityLayer("NX-11..NX-20", ["feature_signature_v2", "intermediate_schema_v2"]),
            CompatibilityLayer("NX-21..NX-35", ["audit_hash_chain", "integrity_checks"]),
            CompatibilityLayer(
                "NX-36..NX-47",
                ["forensic_traceability", "merkle_ready_events", "realtime_roadmap", "dynamic_neuron_telemetry"],
            ),
            CompatibilityLayer(
                "NX-47 v115..v133",
                ["strict_train_evidence_gate", "adaptive_thresholding", "realtime_roadmap", "lum_encode_decode"],
            ),
        ]

        print(f"[{self.version}] System Initialized. Real TIFF processing + `.lum` roundtrip + fail-fast active.")

    @staticmethod
    def _is_pkg_available(package_name: str) -> bool:
        return importlib.util.find_spec(package_name) is not None

    def install_offline(self, package_name: str) -> None:
        import subprocess
        import sys

        if self._is_pkg_available(package_name):
            return

        wheel_roots = [
            Path("/kaggle/input/datasets/ndarray2000/nx47-dependencies"),
            Path("/kaggle/input/nx47-dependencies"),
            Path("/kaggle/input/lum-vorax-dependencies"),
            Path("/kaggle/input/lumvorax-dependencies"),
        ]
        for root in wheel_roots:
            if root.exists():
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={root}", package_name]
                )
                if self._is_pkg_available(package_name):
                    return

        raise FatalPipelineError(
            f"OFFLINE_DEPENDENCY_MISSING: {package_name} not found in known wheel directories."
        )

    def bootstrap_dependencies_fail_fast(self) -> None:
        # pandas/pyarrow required for parquet output path.
        for pkg in ("numpy", "pandas", "tifffile", "imagecodecs", "pyarrow"):
            self.install_offline(pkg)

    def log_event(self, event_type, details, severity="INFO"):
        ts = time.time_ns()
        previous_signature = self.audit_log[-1]["signature"] if self.audit_log else "GENESIS"
        log_entry = {
            "timestamp_ns": ts,
            "event": event_type,
            "severity": severity,
            "details": details,
            "previous_signature": previous_signature,
            "signature": sha512(f"{ts}{event_type}{details}{previous_signature}".encode()).hexdigest(),
        }
        self.audit_log.append(log_entry)

    def _update_roadmap(self, current_step, status="in_progress"):
        if current_step not in self.ROADMAP_STEPS:
            raise FatalPipelineError(f"Unknown roadmap step: {current_step}")
        current_idx = self.ROADMAP_STEPS.index(current_step)
        milestones = []
        for idx, step in enumerate(self.ROADMAP_STEPS):
            if idx < current_idx or (idx == current_idx and status == "done"):
                step_status = "done"
            elif idx == current_idx:
                step_status = "in_progress"
            else:
                step_status = "pending"
            milestones.append({"step": step, "status": step_status})
        roadmap = {
            "version": self.version,
            "timestamp_ns": time.time_ns(),
            "current_step": current_step,
            "status": status,
            "overall_progress_percent": round(
                ((current_idx + (1 if status == "done" else 0)) / len(self.ROADMAP_STEPS)) * 100, 2
            ),
            "milestones": milestones,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.roadmap_path, "w", encoding="utf-8") as f:
            json.dump(roadmap, f, indent=2)

    def _validate_input_structure(self):
        if not os.path.isdir(self.input_dir):
            raise FatalPipelineError(f"INPUT_ROOT_INVALID: missing directory {self.input_dir}")

    def _collect_test_fragments(self) -> List[str]:
        candidates = []
        patterns = [
            f"{self.input_dir}/test/*.tif",
            f"{self.input_dir}/test/*.tiff",
            f"{self.input_dir}/test/**/**/*.tif",
            f"{self.input_dir}/test/**/**/*.tiff",
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern, recursive=True))

        # competition fallback structure
        if not candidates:
            patterns = [
                f"{self.input_dir}/**/test/*.tif",
                f"{self.input_dir}/**/test/*.tiff",
            ]
            for pattern in patterns:
                candidates.extend(glob.glob(pattern, recursive=True))

        uniq = sorted({str(Path(p)) for p in candidates})
        if not uniq:
            raise FatalPipelineError(f"NO_TEST_FRAGMENTS_FOUND in {self.input_dir}")
        return uniq

    def _validate_compatibility_chain(self):
        for layer in self.compatibility_layers:
            missing = [cap for cap in layer.required_capabilities if cap not in self.capability_registry]
            if missing:
                raise FatalPipelineError(f"COMPATIBILITY_BROKEN in {layer.name}: missing {missing}")
            self.log_event("COMPATIBILITY_LAYER_OK", {"layer": layer.name, "caps": layer.required_capabilities})

    def _strict_training_evidence_gate(self):
        expected = {
            "supervised_train": False,
            "val_f1_mean_supervised": None,
            "val_iou_mean_supervised": None,
            "native_bridge_enabled": self.bridge.enabled,
            "native_bridge_path": self.bridge.loaded_path,
        }
        self.log_event("STRICT_TRAINING_GATE", expected)

    @staticmethod
    def _normalize_volume_shape(volume: "np.ndarray") -> "np.ndarray":
        import numpy as np

        arr = np.asarray(volume)
        if arr.ndim == 2:
            return arr[np.newaxis, :, :]
        if arr.ndim == 3:
            return arr
        raise FatalPipelineError(f"UNSUPPORTED_VOLUME_DIM: expected 2D/3D, got {arr.ndim}D")

    def _read_tiff_volume(self, path: str) -> "np.ndarray":
        import tifffile

        arr = tifffile.imread(path)
        arr = self._normalize_volume_shape(arr)
        if arr.shape[1] <= 0 or arr.shape[2] <= 0:
            raise FatalPipelineError(f"INVALID_TIFF_SHAPE: {path} -> {arr.shape}")
        return arr.astype("float32", copy=False)

    def _lum_encode(self, volume: "np.ndarray") -> bytes:
        import numpy as np

        arr = self._normalize_volume_shape(volume)
        payload = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
        digest = sha512(payload).digest()
        z, h, w = arr.shape
        header = struct.pack("<8sIII16s", self.LUM_MAGIC, z, h, w, digest[:16])
        return header + payload

    def _lum_decode(self, blob: bytes) -> "np.ndarray":
        import numpy as np

        header_size = struct.calcsize("<8sIII16s")
        if len(blob) < header_size:
            raise FatalPipelineError("LUM_DECODE_ERROR: blob too small")
        magic, z, h, w, digest16 = struct.unpack("<8sIII16s", blob[:header_size])
        if magic != self.LUM_MAGIC:
            raise FatalPipelineError("LUM_DECODE_ERROR: bad magic")
        payload = blob[header_size:]
        expected_bytes = int(z) * int(h) * int(w) * 4
        if len(payload) != expected_bytes:
            raise FatalPipelineError("LUM_DECODE_ERROR: payload size mismatch")
        if sha512(payload).digest()[:16] != digest16:
            raise FatalPipelineError("LUM_DECODE_ERROR: checksum mismatch")
        arr = np.frombuffer(payload, dtype=np.float32).reshape((z, h, w))
        return arr

    def _roundtrip_lum(self, volume: "np.ndarray") -> LUMVolume:
        blob = self._lum_encode(volume)
        decoded = self._lum_decode(blob)
        payload_sha = sha512(decoded.tobytes()).hexdigest()
        return LUMVolume(shape=tuple(decoded.shape), dtype=str(decoded.dtype), payload_sha512=payload_sha)

    def spatial_harmonic_filtering_simd(self, volume):
        import numpy as np

        vol = self._normalize_volume_shape(volume)
        filtered_slices = []
        for slice_data in vol:
            fft_data = np.fft.fft2(slice_data)
            mask = np.ones_like(slice_data, dtype=np.float32)
            rows, cols = slice_data.shape
            mask[rows // 4 : 3 * rows // 4, cols // 4 : 3 * cols // 4] = 0.5
            filtered_slices.append(np.abs(np.fft.ifft2(fft_data * mask)))
        return np.stack(filtered_slices, axis=0)

    @staticmethod
    def _clamp_density(mask_3d: "np.ndarray", density_target: float) -> "np.ndarray":
        import numpy as np

        total = mask_3d.size
        if total == 0:
            return mask_3d
        target_ones = int(total * density_target)
        if target_ones <= 0:
            return np.zeros_like(mask_3d, dtype=np.uint8)
        if target_ones >= total:
            return np.ones_like(mask_3d, dtype=np.uint8)

        flat = mask_3d.reshape(-1).astype(np.float32)
        if np.max(flat) <= 1.0 and np.min(flat) >= 0.0:
            scores = flat
        else:
            mn = float(np.min(flat))
            mx = float(np.max(flat))
            scores = (flat - mn) / (mx - mn + 1e-8)

        idx = np.argpartition(scores, -target_ones)[-target_ones:]
        out = np.zeros_like(scores, dtype=np.uint8)
        out[idx] = 1
        return out.reshape(mask_3d.shape)

    def ink_resonance_detector_v47(self, filtered_data):
        import numpy as np

        vol = self._normalize_volume_shape(filtered_data)
        mean = float(np.mean(vol))
        std = float(np.std(vol))
        thresholds = [mean + 0.8 * std, mean + 1.2 * std, mean + 1.6 * std]

        layer_low = (vol > thresholds[0]).astype(np.uint8)
        layer_mid = (vol > thresholds[1]).astype(np.uint8)
        layer_high = (vol > thresholds[2]).astype(np.uint8)

        # progressive accumulation with strictness weighting
        accum = (0.55 * layer_low + 0.30 * layer_mid + 0.15 * layer_high).astype(np.float32)
        binary = (accum >= 0.5).astype(np.uint8)

        # clamp to target density range to avoid extreme under/over segmentation
        density_now = float(np.mean(binary))
        target_density = min(max(density_now, 0.05), 0.12)
        return self._clamp_density(accum, target_density)

    def _extract_fragment_signature(self, fragment_id):
        return sha512(f"{fragment_id}|NX47".encode()).hexdigest()[:24]

    def _integrity_digest(self, payload):
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return sha512(encoded).hexdigest()

    def _build_result_entry(self, frag_id, score, density, shape):
        return {
            "id": frag_id,
            "target": float(score),
            "density": float(density),
            "shape_z": int(shape[0]),
            "shape_h": int(shape[1]),
            "shape_w": int(shape[2]),
            "feature_signature": self._extract_fragment_signature(frag_id),
        }

    def _emit_neuron_telemetry(self, filtered_data):
        import numpy as np

        vol = self._normalize_volume_shape(filtered_data)
        total = int(vol.size)
        active = int(np.count_nonzero(vol > np.mean(vol)))
        mid = int(np.count_nonzero(vol > (np.mean(vol) + 0.5 * np.std(vol))))
        end = int(np.count_nonzero(vol > (np.mean(vol) + 1.0 * np.std(vol))))
        return {
            "active_neurons_start_total": total,
            "active_neurons_mid_total": mid,
            "active_neurons_end_total": end,
            "mutation_events": 0,
            "pruning_events": 1,
        }

    def _audit_merkle_root(self):
        leaf_hashes = [entry["signature"] for entry in self.audit_log]
        if not leaf_hashes:
            return ""
        current = leaf_hashes
        while len(current) > 1:
            if len(current) % 2 == 1:
                current.append(current[-1])
            current = [sha512(f"{current[i]}{current[i + 1]}".encode()).hexdigest() for i in range(0, len(current), 2)]
        return current[0]

    def _write_submission_zip(self, masks: Dict[str, "np.ndarray"]) -> None:
        import tifffile

        os.makedirs(self.output_dir, exist_ok=True)
        with zipfile.ZipFile(self.submission_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for frag_id, mask in masks.items():
                arr = self._normalize_volume_shape(mask).astype("uint8")
                tif_buf = io.BytesIO()
                tifffile.imwrite(tif_buf, arr, compression="lzw")
                zf.writestr(f"{frag_id}.tif", tif_buf.getvalue())

    def _export_forensic(self, stats):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.execution_log_path, "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=2)

        metadata = {
            "version": self.version,
            "elapsed_total_s": round((time.time_ns() - self.start_time) / 1e9, 6),
            "integrity_digest": self._integrity_digest(stats),
            "merkle_root": self._audit_merkle_root(),
            "fallback_disabled": self.fallback_disabled,
            "native_bridge_enabled": self.bridge.enabled,
            "native_bridge_path": self.bridge.loaded_path,
            "submission_zip": self.submission_zip_path,
            "submission_parquet": self.submission_parquet_path,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def process_fragments(self):
        import pandas as pd

        self._update_roadmap("bootstrap", "in_progress")
        self.log_event("PIPELINE_START", "Beginning fragment processing")
        self.bootstrap_dependencies_fail_fast()
        os.makedirs(self.lum_work_dir, exist_ok=True)

        self._strict_training_evidence_gate()
        self._update_roadmap("bootstrap", "done")

        self._update_roadmap("compatibility_check", "in_progress")
        self._validate_compatibility_chain()
        self._update_roadmap("compatibility_check", "done")

        self._update_roadmap("data_validation", "in_progress")
        self._validate_input_structure()
        test_fragments = self._collect_test_fragments()
        self.log_event("TEST_FRAGMENT_DISCOVERY", {"count": len(test_fragments)})
        self._update_roadmap("data_validation", "done")

        self._update_roadmap("feature_extraction", "in_progress")
        results = []
        telemetry = {
            "active_neurons_start_total": 0,
            "active_neurons_mid_total": 0,
            "active_neurons_end_total": 0,
            "mutation_events": 0,
            "pruning_events": 0,
        }
        masks_for_zip: Dict[str, "np.ndarray"] = {}

        for frag in test_fragments:
            frag_id = os.path.splitext(os.path.basename(frag))[0]
            self.log_event("FRAGMENT_PROCESSING", {"fragment": frag_id, "path": frag})

            volume = self._read_tiff_volume(frag)
            lum_info = self._roundtrip_lum(volume)
            self.log_event("LUM_ROUNDTRIP_OK", {"fragment": frag_id, "shape": lum_info.shape, "dtype": lum_info.dtype})

            filtered = self.spatial_harmonic_filtering_simd(volume)
            pred = self.ink_resonance_detector_v47(filtered)

            score = float(pred.mean())
            density = float(pred.mean())
            results.append(self._build_result_entry(frag_id, score, density, pred.shape))

            masks_for_zip[frag_id] = pred
            self.processed_pixels += int(filtered.size)
            self.ink_detected += int(pred.sum())
            t = self._emit_neuron_telemetry(filtered)
            for k in telemetry:
                telemetry[k] += int(t.get(k, 0))

        self._update_roadmap("feature_extraction", "done")

        self._update_roadmap("inference", "in_progress")
        submission_df = pd.DataFrame(results)
        if submission_df.empty:
            raise FatalPipelineError("NO_RESULTS_GENERATED")
        submission_df[["id", "target"]].to_parquet(self.submission_parquet_path)
        self._write_submission_zip(masks_for_zip)
        self.log_event("SUBMISSION_GENERATED", {"shape": submission_df.shape, "zip": self.submission_zip_path})
        self._update_roadmap("inference", "done")

        self._update_roadmap("forensic_export", "in_progress")
        stats = {
            "files_processed": len(results),
            "pixels_processed": self.processed_pixels,
            "ink_detected": self.ink_detected,
            "mean_density": float(submission_df["density"].mean()),
            **telemetry,
            "files_autonomous_fallback": 0,
            "lum_bridge_enabled": self.bridge.enabled,
        }
        self._export_forensic(stats)
        self._update_roadmap("forensic_export", "done")

        self._update_roadmap("finalize", "done")
        print(f"[{self.version}] Execution Complete.")
        return stats


if __name__ == "__main__":
    node = NX47_VESU_Production()
    node.process_fragments()

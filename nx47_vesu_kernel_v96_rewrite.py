# ================================================================
# 01) NX47 V113 Kernel
# 02) Kaggle Vesuvius pipeline: discovery -> load -> features -> segment -> overlay -> package
# 03) Robust offline dependencies + LZW-safe TIFF I/O + slice-wise adaptive fusion
# ================================================================

import gc
import importlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
except Exception:
    cp = None
    gpu_gaussian_filter = None

from scipy.ndimage import gaussian_filter as cpu_gaussian_filter
from scipy.ndimage import sobel

import tifffile

try:
    from PIL import Image, ImageSequence
except Exception:
    Image = None
    ImageSequence = None


def install_offline(package_name: str) -> None:
    """Install a package from exact Kaggle offline dependency locations only."""
    exact_wheel_dir = Path("/kaggle/input/datasets/ndarray2000/nx47-dependencies")
    fallback_wheel_dir = Path("/kaggle/input/nx47-dependencies")

    exact_wheels = {
        "imagecodecs": exact_wheel_dir / "imagecodecs-2026.1.14-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "numpy": exact_wheel_dir / "numpy-2.4.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "tifffile": exact_wheel_dir / "tifffile-2026.1.28-py3-none-any.whl",
    }

    # Avoid forcing an incompatible NumPy wheel (e.g. cp311 wheel on cp312 runtime).
    if package_name == "numpy":
        try:
            import numpy as _np  # noqa: F401
            return
        except Exception:
            pass

    if package_name in exact_wheels and exact_wheels[package_name].exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", str(exact_wheels[package_name])])
            return
        except subprocess.CalledProcessError:
            # Fall back to --find-links resolution for interpreter/platform compatibility.
            pass

    if exact_wheel_dir.exists():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={exact_wheel_dir}",
                package_name,
            ]
        )
        return

    if fallback_wheel_dir.exists():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={fallback_wheel_dir}",
                package_name,
            ]
        )
        return

    raise RuntimeError(
        "Offline dependency directory not found. Expected one of: "
        f"{exact_wheel_dir} or {fallback_wheel_dir}"
    )


def bootstrap_dependencies_fail_fast() -> None:
    # Respect exact offline process and ordering requested for Kaggle runtime.
    # NumPy is often already present and wheel tags may differ by Python minor version.
    install_offline("numpy")
    install_offline("imagecodecs")
    install_offline("tifffile")

    # Refresh tifffile module state after wheel installs in the same interpreter.
    global tifffile
    tifffile = importlib.reload(tifffile)


def ensure_imagecodecs() -> bool:
    """Ensure imagecodecs is available for LZW TIFF read/write."""
    try:
        import imagecodecs  # noqa: F401
        return True
    except Exception:
        pass

    try:
        install_offline("imagecodecs")
        import imagecodecs  # noqa: F401

        # tifffile may have been imported before imagecodecs installation.
        # Reload it so compression codecs are re-detected in the same process.
        global tifffile
        tifffile = importlib.reload(tifffile)
        return True
    except Exception:
        return False


def read_tiff_lzw_safe(path: Path) -> np.ndarray:
    """Read TIFF volumes with tifffile, then robustly fallback if LZW codecs still fail."""
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        if "requires the 'imagecodecs' package" not in str(exc):
            raise

    # Try to install/refresh codecs, then retry tifffile once.
    ensure_imagecodecs()
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        if "requires the 'imagecodecs' package" not in str(exc):
            raise

    # Final fallback: Pillow decoder path.
    if Image is None or ImageSequence is None:
        raise RuntimeError(
            "LZW TIFF read failed after imagecodecs bootstrap and Pillow fallback is unavailable."
        )

    with Image.open(path) as img:
        frames = [np.array(frame, dtype=np.float32) for frame in ImageSequence.Iterator(img)]

    if not frames:
        raise RuntimeError(f"No frames decoded from TIFF: {path}")

    return np.stack(frames, axis=0)


def write_tiff_lzw_safe(path: Path, arr: np.ndarray) -> None:
    """Write LZW TIFF using tifffile, fallback to Pillow when codecs are unavailable."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported TIFF array shape for write: {arr.shape}")

    try:
        if ensure_imagecodecs():
            tifffile.imwrite(path, arr, compression="LZW")
            return
    except Exception:
        pass

    if Image is None:
        raise RuntimeError(
            "LZW TIFF write failed: imagecodecs unavailable and Pillow fallback unavailable."
        )

    pages = [Image.fromarray(frame.astype(np.uint8)) for frame in arr]
    if not pages:
        raise RuntimeError("Cannot write empty TIFF volume")
    pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_lzw")


@dataclass
class PlanStep:
    name: str
    description: str
    progress: float = 0.0
    done: bool = False


@dataclass
class PlanTracker:
    output_path: Path
    steps: List[PlanStep] = field(default_factory=list)

    def add_step(self, name: str, description: str) -> None:
        self.steps.append(PlanStep(name=name, description=description))

    def update(self, name: str, progress: float, done: bool = False) -> None:
        for step in self.steps:
            if step.name == name:
                step.progress = float(np.clip(progress, 0.0, 100.0))
                step.done = done
                break
        self._write()

    def _write(self) -> None:
        payload = {
            "generated_at_ns": time.time_ns(),
            "roadmap": [
                {
                    "name": s.name,
                    "description": s.description,
                    "progress_percent": round(s.progress, 2),
                    "done": s.done,
                }
                for s in self.steps
            ],
            "overall_progress_percent": round(float(np.mean([s.progress for s in self.steps])) if self.steps else 0.0, 2),
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class NX47V113Kernel:
    """
    V113 pipeline data-driven for Vesuvius test_images format.
    - Input format: *.tif volume files in /kaggle/input/competitions/vesuvius-challenge-surface-detection/test_images
    - Output format: submission.zip containing one LZW-compressed TIFF mask per input file with same filename.
    """

    def __init__(
        self,
        root: Path = Path("/kaggle/input/competitions/vesuvius-challenge-surface-detection"),
        output_dir: Path = Path("/kaggle/working"),
        overlay_stride: int = 8,
    ) -> None:
        self.root = root
        self.test_dir = self.root / "test_images"
        self.output_dir = output_dir
        self.tmp_dir = output_dir / "tmp_masks"
        self.overlay_dir = output_dir / "overlays"
        self.roadmap_path = output_dir / "v113_roadmap_realtime.json"
        self.submission_path = output_dir / "submission.zip"
        self.overlay_stride = max(1, int(overlay_stride))

        bootstrap_dependencies_fail_fast()
        if not ensure_imagecodecs():
            raise RuntimeError(
                "imagecodecs is mandatory at kernel startup for LZW TIFF I/O. "
                "Provide offline wheels in /kaggle/input/datasets/ndarray2000/nx47-dependencies (preferred) or /kaggle/input/nx47-dependencies."
            )

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

        self.plan = PlanTracker(output_path=self.roadmap_path)
        self.plan.add_step("discovery", "Validation des dossiers/format de competition")
        self.plan.add_step("load", "Chargement volume TIFF + normalisation")
        self.plan.add_step("features", "Extraction NX-47 / Neurones atom / NX-47 fusion")
        self.plan.add_step("segment", "Segmentation dynamique slice par slice")
        self.plan.add_step("overlay", "Generation overlay pixel/motifs + reconstruction")
        self.plan.add_step("package", "Generation submission.zip format exact Kaggle")
        self.plan._write()

        self.logs: List[Dict[str, object]] = []

    def log(self, event: str, **kwargs: object) -> None:
        record = {"ts_ns": time.time_ns(), "event": event, **kwargs}
        self.logs.append(record)
        print(json.dumps(record, ensure_ascii=False))

    @property
    def using_gpu(self) -> bool:
        return cp is not None

    def _xp(self):
        return cp if self.using_gpu else np

    def _gaussian(self, arr, sigma: float):
        if self.using_gpu:
            return gpu_gaussian_filter(arr, sigma=sigma)
        return cpu_gaussian_filter(arr, sigma=sigma)

    def discover_inputs(self) -> List[Path]:
        self.plan.update("discovery", 25.0)
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Missing test_images directory: {self.test_dir}")
        files = sorted(self.test_dir.rglob("*.tif"))
        if not files:
            raise RuntimeError(f"No TIFF files found in {self.test_dir}")
        self.log("INPUT_DISCOVERED", file_count=len(files), using_gpu=self.using_gpu)
        self.plan.update("discovery", 100.0, done=True)
        return files

    def read_volume(self, path: Path) -> np.ndarray:
        self.plan.update("load", 25.0)
        vol = read_tiff_lzw_safe(path).astype(np.float32)
        if vol.ndim != 3:
            raise RuntimeError(f"Unsupported TIFF shape for {path.name}: {vol.shape}")
        vmin = float(vol.min())
        vmax = float(vol.max())
        vol = (vol - vmin) / (vmax - vmin + 1e-6)
        self.plan.update("load", 100.0, done=True)
        return vol

    def extract_features(self, vol: np.ndarray) -> Dict[str, float]:
        self.plan.update("features", 20.0)
        gx = sobel(vol, axis=2)
        gy = sobel(vol, axis=1)
        gz = sobel(vol, axis=0)

        gradient_energy = float(np.mean(np.sqrt(gx * gx + gy * gy + gz * gz)))
        intensity_std = float(np.std(vol))
        intensity_mean = float(np.mean(vol))
        p95 = float(np.percentile(vol, 95))
        p50 = float(np.percentile(vol, 50))

        hist_counts, _ = np.histogram(vol, bins=64, range=(0.0, 1.0), density=False)
        hist_probs = hist_counts.astype(np.float64)
        hist_probs /= max(float(hist_probs.sum()), 1.0)
        entropy = -float(np.sum(hist_probs * np.log(hist_probs + 1e-12)))

        grad_term = math.tanh(gradient_energy / 2.0)
        std_term = math.tanh(intensity_std * 8.0)
        contrast_term = math.tanh((p95 - p50) * 6.0)
        entropy_term = entropy / math.log(64.0)

        nx47_signal = 0.35 * grad_term + 0.30 * std_term + 0.20 * contrast_term + 0.15 * entropy_term
        atom_neuron_signal = 0.5 * (1.0 + math.tanh((intensity_mean - 0.35) * 6.0)) + 0.5 * (1.0 + math.tanh((intensity_std - 0.10) * 10.0))

        fusion_raw = 0.7 * nx47_signal + 0.3 * (atom_neuron_signal - 0.5)
        fusion_score = float(np.clip(0.5 + 0.5 * math.tanh(fusion_raw * 1.2), 0.15, 0.85))

        self.plan.update("features", 100.0, done=True)
        return {
            "gradient_energy": gradient_energy,
            "intensity_std": intensity_std,
            "intensity_mean": intensity_mean,
            "entropy": entropy,
            "nx47_signal": nx47_signal,
            "atom_neuron_signal": atom_neuron_signal,
            "fusion_score": fusion_score,
        }

    def _slice_fusion_score(self, vol: np.ndarray, z: int, global_fusion: float) -> float:
        """Compute slice-aware fusion score to avoid constant global fusion collapse."""
        z_count = vol.shape[0]
        z0, z1 = max(0, z - 2), min(z_count, z + 3)
        local = vol[z0:z1]

        local_std = float(np.std(local))
        local_mean = float(np.mean(local))
        local_p95 = float(np.percentile(local, 95))
        local_p50 = float(np.percentile(local, 50))

        # Use lightweight directional gradients on current slice.
        sl = vol[z]
        gx = np.diff(sl, axis=1, append=sl[:, -1:])
        gy = np.diff(sl, axis=0, append=sl[-1:, :])
        grad_energy = float(np.mean(np.sqrt(gx * gx + gy * gy)))

        slice_signal = (
            0.35 * math.tanh(grad_energy * 4.0)
            + 0.30 * math.tanh(local_std * 8.0)
            + 0.20 * math.tanh((local_p95 - local_p50) * 8.0)
            + 0.15 * math.tanh((local_mean - 0.30) * 6.0)
        )
        slice_fusion = float(np.clip(0.5 + 0.5 * math.tanh(slice_signal * 2.0), 0.05, 0.95))

        # Blend global + local so each slice can vary while keeping stability.
        return float(np.clip(0.65 * global_fusion + 0.35 * slice_fusion, 0.05, 0.95))

    def segment_volume(self, vol: np.ndarray, fusion_score: float) -> np.ndarray:
        self.plan.update("segment", 10.0)
        xp = self._xp()
        vol_xp = xp.asarray(vol)

        sigma = float(max(0.5, np.std(vol) * 1.1 + 0.35))
        smooth_lo = self._gaussian(vol_xp, sigma=sigma)
        smooth_hi = self._gaussian(vol_xp, sigma=sigma * 1.8)
        smooth = 0.65 * smooth_lo + 0.35 * smooth_hi
        resid = vol_xp - smooth

        z_count = int(vol.shape[0])
        masks = []
        for z in range(z_count):
            if z % max(1, z_count // 20) == 0:
                self.plan.update("segment", 10.0 + 80.0 * (z / max(1, z_count - 1)))

            z0, z1 = max(0, z - 1), min(z_count, z + 2)
            proj = xp.mean(resid[z0:z1], axis=0)
            local_std = float(xp.std(vol_xp[max(0, z - 2):min(z_count, z + 3)]))

            slice_fusion = self._slice_fusion_score(vol, z, fusion_score)
            adaptive_weight_raw = 0.12 + 0.22 * math.tanh(slice_fusion * 2.2) * math.tanh(local_std * 6.0)
            adaptive_cap = 0.238 + 0.018 * math.tanh((local_std - 0.108) * 18.0)
            adaptive_weight = min(adaptive_weight_raw, adaptive_cap)
            proj = proj * (1.0 + adaptive_weight * 0.6)

            proj_cpu = cp.asnumpy(proj) if self.using_gpu else proj
            p_lo = float(np.percentile(proj_cpu, 75))
            p_hi = float(np.percentile(proj_cpu, 95))

            den = (p_hi - p_lo) + 1e-6
            w = xp.clip((proj - p_lo) / den, 0.0, 1.0)
            hi = proj > p_hi
            lo = proj > p_lo
            final = (w * xp.logical_and(hi, lo) + (1.0 - w) * xp.logical_or(hi, lo)) > 0.5

            final_cpu = cp.asnumpy(final) if self.using_gpu else np.asarray(final)
            masks.append(final_cpu.astype(np.uint8) * 255)

            self.log(
                "SLICE_METRIC",
                z=z,
                fusion_score=round(slice_fusion, 6),
                local_std=round(local_std, 6),
                adaptive_weight=round(adaptive_weight, 6),
                p_lo=round(p_lo, 6),
                p_hi=round(p_hi, 6),
            )

        stacked = np.stack(masks)
        self.plan.update("segment", 100.0, done=True)
        return stacked

    def _build_submission_mask(self, mask_stack: np.ndarray, file_name: str) -> np.ndarray:
        """Convert 3D mask stack to strict binary 2D mask (values 0/1) for Kaggle submission."""
        reconstruction = np.mean(mask_stack.astype(np.float32) / 255.0, axis=0)

        p01 = float(np.percentile(reconstruction, 1))
        p05 = float(np.percentile(reconstruction, 5))
        p50 = float(np.percentile(reconstruction, 50))
        p95 = float(np.percentile(reconstruction, 95))
        p99 = float(np.percentile(reconstruction, 99))

        env_threshold = os.environ.get(
            "V113_SUBMISSION_THRESHOLD",
            os.environ.get(
                "V112_SUBMISSION_THRESHOLD",
                os.environ.get("V111_SUBMISSION_THRESHOLD", os.environ.get("V110_SUBMISSION_THRESHOLD", os.environ.get("V109_SUBMISSION_THRESHOLD"))),
            ),
        )
        if env_threshold is not None:
            threshold = float(np.clip(float(env_threshold), 0.0, 1.0))
            threshold_source = "env"
        else:
            target_active_ratio = float(np.clip(float(os.environ.get("V113_TARGET_ACTIVE_RATIO", "0.20")), 0.05, 0.35))
            low = float(np.percentile(reconstruction, 40))
            high = float(np.percentile(reconstruction, 95))
            if high <= low + 1e-9:
                threshold = float(np.percentile(reconstruction, 80))
                threshold_source = "fallback_percentile80"
            else:
                candidates = np.linspace(low, high, 64, dtype=np.float32)
                ratios = np.array([float((reconstruction >= th).mean()) for th in candidates], dtype=np.float32)
                idx = int(np.argmin(np.abs(ratios - target_active_ratio)))
                threshold = float(candidates[idx])
                threshold_source = f"target_active_ratio_{target_active_ratio:.3f}"

        sweep_thresholds = np.linspace(float(np.percentile(reconstruction, 50)), float(np.percentile(reconstruction, 95)), 21, dtype=np.float32)
        sweep = {f"{th:.4f}": round(float((reconstruction >= th).mean()), 6) for th in sweep_thresholds}

        binary_mask = (reconstruction >= threshold).astype(np.uint8)
        active_ratio = float(binary_mask.mean())

        self.log(
            "SUBMISSION_MASK_STATS",
            version="v113",
            file=file_name,
            threshold=round(threshold, 6),
            threshold_source=threshold_source,
            active_ratio=round(active_ratio, 6),
            recon_min=round(float(reconstruction.min()), 6),
            recon_max=round(float(reconstruction.max()), 6),
            recon_mean=round(float(reconstruction.mean()), 6),
            recon_std=round(float(reconstruction.std()), 6),
            p01=round(p01, 6),
            p05=round(p05, 6),
            p50=round(p50, 6),
            p95=round(p95, 6),
            p99=round(p99, 6),
            min=int(binary_mask.min()),
            max=int(binary_mask.max()),
            sweep_active_ratio=sweep,
        )

        if binary_mask.max() > 1 or binary_mask.min() < 0:
            raise RuntimeError("Submission mask must be strictly binary uint8 in {0,1}.")

        return binary_mask

    def _save_overlay(self, file_stem: str, vol: np.ndarray, mask: np.ndarray) -> None:
        self.plan.update("overlay", 20.0)
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.log("OVERLAY_SKIPPED", reason=f"matplotlib unavailable: {exc}")
            self.plan.update("overlay", 100.0, done=True)
            return

        z_count = vol.shape[0]
        for idx, z in enumerate(range(0, z_count, self.overlay_stride), start=1):
            base = vol[z]
            m = (mask[z] > 0).astype(np.uint8)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(base, cmap="gray")
            ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45)
            ax.set_title(f"{file_stem} | slice {z} | legende: orange=motifs detectes")
            ax.axis("off")
            out_path = self.overlay_dir / f"{file_stem}_slice_{z:04d}_overlay.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            prog = 20.0 + 70.0 * (idx / max(1, len(range(0, z_count, self.overlay_stride))))
            self.plan.update("overlay", prog)

        recon = (np.mean(mask.astype(np.float32) / 255.0, axis=0) * 255.0).astype(np.uint8)
        recon_path = self.overlay_dir / f"{file_stem}_reconstruction_total.png"
        tifffile.imwrite(self.overlay_dir / f"{file_stem}_reconstruction_total.tif", recon)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(recon, cmap="inferno")
        ax.set_title(f"{file_stem} | reconstruction totale pixel par pixel")
        ax.axis("off")
        fig.savefig(recon_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        self.plan.update("overlay", 100.0, done=True)

    def run(self) -> Path:
        files = self.discover_inputs()

        self.plan.update("package", 10.0)
        with zipfile.ZipFile(self.submission_path, "w", zipfile.ZIP_STORED) as zf:
            for i, fpath in enumerate(files, start=1):
                self.log("FILE_START", file=fpath.name, index=i, total=len(files))
                self.plan.update("load", 0.0, done=False)
                self.plan.update("features", 0.0, done=False)
                self.plan.update("segment", 0.0, done=False)
                self.plan.update("overlay", 0.0, done=False)

                vol = self.read_volume(fpath)
                features = self.extract_features(vol)
                self.log("FEATURES", file=fpath.name, **{k: round(v, 6) for k, v in features.items()})

                mask_stack = self.segment_volume(vol, fusion_score=features["fusion_score"])
                self._save_overlay(fpath.stem, vol, mask_stack)

                submission_mask = self._build_submission_mask(mask_stack, fpath.name)
                out_mask = self.tmp_dir / fpath.name
                write_tiff_lzw_safe(out_mask, submission_mask[np.newaxis, ...])
                zf.write(out_mask, arcname=fpath.name)
                out_mask.unlink(missing_ok=True)
                gc.collect()

                self.log("FILE_DONE", file=fpath.name, slices=int(mask_stack.shape[0]))
                self.plan.update("package", 10.0 + 85.0 * (i / len(files)))

        metadata = {
            "submission_zip": str(self.submission_path),
            "input_dir": str(self.test_dir),
            "output_masks_format": "LZW compressed TIFF, binary uint8 values in {0,1}, same name as input",
            "overlay_dir": str(self.overlay_dir),
            "log_count": len(self.logs),
            "gpu": self.using_gpu,
        }
        (self.output_dir / "v113_execution_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.output_dir / "v113_execution_logs.json").write_text(json.dumps(self.logs, indent=2), encoding="utf-8")
        self.plan.update("package", 100.0, done=True)
        self.log("EXEC_COMPLETE", submission=str(self.submission_path))
        return self.submission_path


# Backward-compatible class aliases (legacy references).
NX47V112Kernel = NX47V113Kernel
NX47V111Kernel = NX47V113Kernel
NX47V110Kernel = NX47V113Kernel
NX47V109Kernel = NX47V113Kernel
NX47V108Kernel = NX47V113Kernel
NX47V107Kernel = NX47V113Kernel
NX47V106Kernel = NX47V113Kernel
NX47V96Kernel = NX47V113Kernel


if __name__ == "__main__":
    kernel = NX47V113Kernel(
        root=Path(os.environ.get("VESUVIUS_ROOT", "/kaggle/input/competitions/vesuvius-challenge-surface-detection")),
        output_dir=Path(os.environ.get("VESUVIUS_OUTPUT", "/kaggle/working")),
        overlay_stride=int(
            os.environ.get("V113_OVERLAY_STRIDE", os.environ.get("V112_OVERLAY_STRIDE", os.environ.get("V111_OVERLAY_STRIDE", os.environ.get("V110_OVERLAY_STRIDE", os.environ.get("V109_OVERLAY_STRIDE", os.environ.get("V108_OVERLAY_STRIDE", os.environ.get("V107_OVERLAY_STRIDE", os.environ.get("V106_OVERLAY_STRIDE", os.environ.get("V96_OVERLAY_STRIDE", "8")))))))))
        ),
    )
    submission = kernel.run()
    print(f"READY: {submission}")

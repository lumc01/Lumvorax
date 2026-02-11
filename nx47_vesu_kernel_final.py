import gc
import json
import math
import os
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


class NX47V96Kernel:
    """
    V96 pipeline data-driven for Vesuvius test_images format.
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
        self.roadmap_path = output_dir / "v96_roadmap_realtime.json"
        self.submission_path = output_dir / "submission.zip"
        self.overlay_stride = max(1, int(overlay_stride))

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
        vol = tifffile.imread(path).astype(np.float32)
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

        hist, _ = np.histogram(vol, bins=64, range=(0.0, 1.0), density=True)
        entropy = -float(np.sum(hist * np.log(hist + 1e-8)))

        nx47_signal = 0.35 * gradient_energy + 0.30 * intensity_std + 0.20 * (p95 - p50) + 0.15 * entropy
        atom_neuron_signal = (1.0 + math.tanh((intensity_mean - 0.45) * 4.0)) * (1.0 + math.tanh((intensity_std - 0.12) * 8.0))
        fusion_score = 0.7 * math.tanh(nx47_signal * 3.0) + 0.3 * math.tanh(atom_neuron_signal)
        fusion_score = float(np.clip(fusion_score + 0.05, 0.0, 1.0))

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

    def segment_volume(self, vol: np.ndarray, fusion_score: float) -> np.ndarray:
        self.plan.update("segment", 10.0)
        xp = self._xp()
        vol_xp = xp.asarray(vol)

        sigma = float(max(0.5, np.std(vol) * 1.1 + 0.35))
        smooth = self._gaussian(vol_xp, sigma=sigma)
        resid = vol_xp - smooth

        z_count = int(vol.shape[0])
        masks = []
        for z in range(z_count):
            if z % max(1, z_count // 20) == 0:
                self.plan.update("segment", 10.0 + 80.0 * (z / max(1, z_count - 1)))

            z0, z1 = max(0, z - 1), min(z_count, z + 2)
            proj = xp.mean(resid[z0:z1], axis=0)
            local_std = float(xp.std(vol_xp[max(0, z - 2):min(z_count, z + 3)]))

            adaptive_weight = 0.12 + 0.22 * math.tanh(fusion_score * 2.2) * math.tanh(local_std * 6.0)
            proj = proj + adaptive_weight

            proj_cpu = cp.asnumpy(proj) if self.using_gpu else proj
            p_lo = float(np.percentile(proj_cpu, 84))
            p_hi = float(np.percentile(proj_cpu, 92))

            den = (p_hi - p_lo) + 1e-6
            w = xp.clip((proj - p_lo) / den, 0.0, 1.0)
            hi = proj > p_hi
            lo = proj > p_lo
            final = (w * xp.logical_and(hi, lo) + (1.0 - w) * xp.logical_or(hi, lo)) > 0.5
            masks.append(final.astype(xp.uint8) * 255)

            self.log(
                "SLICE_METRIC",
                z=z,
                fusion_score=round(fusion_score, 6),
                local_std=round(local_std, 6),
                adaptive_weight=round(adaptive_weight, 6),
                p_lo=round(p_lo, 6),
                p_hi=round(p_hi, 6),
            )

        stacked = xp.stack(masks)
        self.plan.update("segment", 100.0, done=True)
        return cp.asnumpy(stacked) if self.using_gpu else stacked

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

                mask = self.segment_volume(vol, fusion_score=features["fusion_score"])
                self._save_overlay(fpath.stem, vol, mask)

                out_mask = self.tmp_dir / fpath.name
                tifffile.imwrite(out_mask, mask, compression="LZW")
                zf.write(out_mask, arcname=fpath.name)
                out_mask.unlink(missing_ok=True)
                gc.collect()

                self.log("FILE_DONE", file=fpath.name, slices=int(mask.shape[0]))
                self.plan.update("package", 10.0 + 85.0 * (i / len(files)))

        metadata = {
            "submission_zip": str(self.submission_path),
            "input_dir": str(self.test_dir),
            "output_masks_format": "LZW compressed TIFF, same name as input",
            "overlay_dir": str(self.overlay_dir),
            "log_count": len(self.logs),
            "gpu": self.using_gpu,
        }
        (self.output_dir / "v96_execution_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.output_dir / "v96_execution_logs.json").write_text(json.dumps(self.logs, indent=2), encoding="utf-8")
        self.plan.update("package", 100.0, done=True)
        self.log("EXEC_COMPLETE", submission=str(self.submission_path))
        return self.submission_path


if __name__ == "__main__":
    kernel = NX47V96Kernel(
        root=Path(os.environ.get("VESUVIUS_ROOT", "/kaggle/input/competitions/vesuvius-challenge-surface-detection")),
        output_dir=Path(os.environ.get("VESUVIUS_OUTPUT", "/kaggle/working")),
        overlay_stride=int(os.environ.get("V96_OVERLAY_STRIDE", "8")),
    )
    submission = kernel.run()
    print(f"READY: {submission}")

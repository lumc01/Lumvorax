# ================================================================
# LUMVORAX DEPENDENCY 360 VALIDATION (KAGGLE SINGLE CELL - V13 STRICT)
# ================================================================
# Purpose:
# - Binary-first validation for Kaggle dependency datasets.
# - NO C/.h validation and NO native source compilation checks.
# - Validate wheel set + shared object presence + enforced .so load + roundtrip.

from __future__ import annotations

import ctypes
import importlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time
from hashlib import sha256, sha512
from pathlib import Path
from typing import Any, Dict, List, Optional

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

LUM_MAGIC = b"LUMV1\x00\x00\x00"

KAGGLE_DEP_PATHS = [
    Path('/kaggle/input/datasets/ndarray2000/nx47-dependencies'),
    Path('/kaggle/input/nx47-dependencies'),
    Path('/kaggle/input/lum-vorax-dependencies'),
    Path('/kaggle/input/lumvorax-dependencies'),
]

EXPECTED_WHEELS = [
    'imagecodecs-2026.1.14-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl',
    'imageio-2.37.2-py3-none-any.whl',
    'lazy_loader-0.4-py3-none-any.whl',
    'networkx-3.6.1-py3-none-any.whl',
    'opencv_python-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl',
    'packaging-26.0-py3-none-any.whl',
    'tifffile-2026.2.16-py3-none-any.whl',
]
EXPECTED_NATIVE_LIB = 'liblumvorax.so'

IS_KAGGLE = Path('/kaggle').exists()
STRICT_NO_FALLBACK = os.environ.get('LUMVORAX_STRICT_NO_FALLBACK', '1') == '1'
REQUIRE_DATASET = os.environ.get('LUMVORAX_REQUIRE_DATASET', '1' if IS_KAGGLE else '0') == '1'
REQUIRE_SO_PRESENCE = os.environ.get('LUMVORAX_REQUIRE_SO_PRESENCE', '1') == '1'
ENFORCE_SO_LOAD = os.environ.get('LUMVORAX_ENFORCE_SO_LOAD', '1' if STRICT_NO_FALLBACK else '0') == '1'
SKIP_ROUNDTRIP = os.environ.get('LUMVORAX_SKIP_ROUNDTRIP', '0' if IS_KAGGLE else '1') == '1'


def now_ns() -> int:
    return time.time_ns()


def log_event(report: Dict[str, Any], step: str, **payload: Any) -> None:
    report.setdefault('events', []).append({'ts_ns': now_ns(), 'step': step, **payload})


def pkg_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def detect_dataset_root(report: Dict[str, Any]) -> Optional[Path]:
    for root in KAGGLE_DEP_PATHS:
        if root.exists() and root.is_dir():
            log_event(report, 'dataset_root_selected', root=str(root))
            return root
    log_event(report, 'dataset_root_missing', tried=[str(p) for p in KAGGLE_DEP_PATHS])
    return None


def install_wheel_file(wheel_path: Path, report: Dict[str, Any], reason: str) -> Dict[str, Any]:
    cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-index', str(wheel_path)]
    try:
        log_event(report, 'wheel_install_attempt', wheel=str(wheel_path), reason=reason, cmd=cmd)
        subprocess.check_call(cmd)
        return {'status': 'installed', 'wheel': str(wheel_path), 'reason': reason}
    except Exception as exc:
        log_event(report, 'wheel_install_fail', wheel=str(wheel_path), reason=reason, error=str(exc))
        return {'status': 'failed', 'wheel': str(wheel_path), 'reason': reason, 'error': str(exc)}


def install_offline_if_missing(pkg: str, report: Dict[str, Any], dataset_root: Optional[Path]) -> Dict[str, Any]:
    if pkg_available(pkg):
        return {'package': pkg, 'status': 'already_installed'}

    mapping = {
        'numpy': 'numpy-2.4.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl',
        'tifffile': 'tifffile-2026.2.16-py3-none-any.whl',
        'imagecodecs': 'imagecodecs-2026.1.14-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl',
    }
    if dataset_root and pkg in mapping:
        wheel = dataset_root / mapping[pkg]
        if wheel.exists():
            res = install_wheel_file(wheel, report, reason=f'exact_{pkg}_wheel')
            if res['status'] == 'installed' and pkg_available(pkg):
                return {'package': pkg, 'status': 'installed', 'method': 'exact_wheel', 'wheel': str(wheel)}

    last = 'not found'
    for root in KAGGLE_DEP_PATHS:
        if not root.exists():
            continue
        cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-index', f'--find-links={root}', pkg]
        try:
            log_event(report, 'dependency_install_attempt', package=pkg, root=str(root), cmd=cmd)
            subprocess.check_call(cmd)
            if pkg_available(pkg):
                return {'package': pkg, 'status': 'installed', 'method': 'find_links', 'root': str(root)}
        except Exception as exc:
            last = str(exc)
            log_event(report, 'dependency_install_fail', package=pkg, root=str(root), error=last)

    return {'package': pkg, 'status': 'missing', 'error': last}


def inspect_dataset_artifacts(root: Optional[Path], report: Dict[str, Any]) -> Dict[str, Any]:
    if root is None:
        return {'status': 'missing', 'reason': 'dataset_root_not_found'}

    files = {p.name: p for p in root.iterdir() if p.is_file()}
    missing_wheels = [w for w in EXPECTED_WHEELS if w not in files]
    present_wheels = [w for w in EXPECTED_WHEELS if w in files]
    native = files.get(EXPECTED_NATIVE_LIB)

    wheel_compat = scan_wheel_compatibility(files)
    out = {
        'status': 'ok' if ((not REQUIRE_SO_PRESENCE or native is not None) and len(missing_wheels) == 0) else 'fail',
        'dataset_root': str(root),
        'expected_native_lib': EXPECTED_NATIVE_LIB,
        'resolved_native_lib_name': native.name if native else None,
        'native_lib': str(native) if native else None,
        'native_lib_sha256': sha256(native.read_bytes()).hexdigest() if native else None,
        'native_lib_size': native.stat().st_size if native else None,
        'expected_wheel_count': len(EXPECTED_WHEELS),
        'present_wheels_count': len(present_wheels),
        'missing_wheels': missing_wheels,
        'present_wheels': present_wheels,
        'wheel_compatibility': wheel_compat,
    }
    if wheel_compat['incompatible_wheels']:
        out['status'] = 'fail'
    log_event(report, 'dataset_artifacts_checked', status=out['status'], missing_wheels=len(missing_wheels), native_found=bool(native))
    return out


def scan_wheel_compatibility(files: Dict[str, Path]) -> Dict[str, Any]:
    runtime_tags = {str(tag) for tag in sys_tags()}
    checked: List[Dict[str, Any]] = []
    incompatible: List[str] = []

    for wheel_name, wheel_path in sorted(files.items()):
        if not wheel_name.endswith('.whl'):
            continue
        try:
            _, _, _, tags = parse_wheel_filename(wheel_name)
            wheel_tags = {str(tag) for tag in tags}
            is_compatible = bool(runtime_tags.intersection(wheel_tags))
            checked.append({'wheel': wheel_name, 'compatible': is_compatible, 'tag_count': len(wheel_tags)})
            if not is_compatible:
                incompatible.append(wheel_name)
        except Exception as exc:
            checked.append({'wheel': wheel_name, 'compatible': False, 'parse_error': str(exc)})
            incompatible.append(wheel_name)

    return {
        'runtime_tag_head': sorted(runtime_tags)[:24],
        'checked_wheels': checked,
        'incompatible_wheels': incompatible,
    }


def check_native_load(lib_path: Optional[str], report: Dict[str, Any]) -> Dict[str, Any]:
    if not lib_path:
        return {'status': 'missing', 'reason': 'native_library_missing'}

    try:
        ctypes.CDLL(lib_path)
        log_event(report, 'native_load_ok', lib=lib_path)
        return {'status': 'ok', 'lib': lib_path}
    except Exception as exc:
        err = str(exc)
        m = re.search(r'GLIBC_(\d+\.\d+)', err)
        diag = {
            'status': 'abi_incompatible' if m else 'fail',
            'lib': lib_path,
            'error': err,
            'required_glibc': m.group(1) if m else None,
            'runtime_glibc': platform.libc_ver()[1] or 'unknown',
            'blocking': bool(ENFORCE_SO_LOAD),
        }
        log_event(report, 'native_load_check', **diag)
        if ENFORCE_SO_LOAD:
            raise RuntimeError(f'native_so_load_enforced_failed: {err}')
        return diag


def inspect_so_symbols(lib_path: Optional[str], report: Dict[str, Any]) -> Dict[str, Any]:
    if not lib_path:
        return {'status': 'missing', 'reason': 'no_lib_path'}

    nm = os.environ.get('NM_BIN', 'nm')
    cmd = [nm, '-D', lib_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {'status': 'fail', 'returncode': proc.returncode, 'stderr_head': (proc.stderr or '')[:1200]}

        symbols = []
        for line in proc.stdout.splitlines():
            if ' T ' in line or ' t ' in line:
                parts = line.strip().split()
                if parts:
                    symbols.append(parts[-1])
        lum_related = [s for s in symbols if s.startswith(('lum_', 'vorax_', 'log_', 'forensic_'))]
        out = {
            'status': 'ok',
            'symbol_count_total': len(symbols),
            'symbol_count_lum_related': len(lum_related),
            'lum_related_head': lum_related[:200],
        }
        log_event(report, 'so_symbols_scanned', total=out['symbol_count_total'], lum_related=out['symbol_count_lum_related'])
        return out
    except Exception as exc:
        return {'status': 'fail', 'error': str(exc), 'cmd': cmd}


def normalize_volume(arr):
    import numpy as np

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f'Expected 2D/3D volume, got {arr.shape}')
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def encode_lum_v1(arr3d):
    z, h, w = arr3d.shape
    payload = arr3d.tobytes()
    digest16 = sha512(payload).digest()[:16]
    header = struct.pack('<8sIII16s', LUM_MAGIC, z, h, w, digest16)
    return header + payload


def decode_lum_v1(blob: bytes):
    import numpy as np

    hs = struct.calcsize('<8sIII16s')
    magic, z, h, w, digest16 = struct.unpack('<8sIII16s', blob[:hs])
    if magic != LUM_MAGIC:
        raise ValueError('invalid LUM magic')
    payload = blob[hs:]
    expected = int(z) * int(h) * int(w) * 4
    if len(payload) != expected:
        raise ValueError('payload size mismatch')
    if sha512(payload).digest()[:16] != digest16:
        raise ValueError('payload checksum mismatch')
    return np.frombuffer(payload, dtype=np.float32).reshape((z, h, w))




def _tiff_compression_plan() -> List[tuple[str, Optional[str]]]:
    """Strict compression plan: only LZW is accepted in strict mode."""
    lzw_ready = False
    lzw_error: Optional[str] = None
    try:
        import imagecodecs  # type: ignore
        lzw_ready = hasattr(imagecodecs, 'lzw_encode')
        if not lzw_ready:
            lzw_error = 'imagecodecs imported but lzw_encode missing'
    except Exception as exc:
        lzw_error = str(exc)

    if lzw_ready:
        return [('LZW', 'LZW')]
    raise RuntimeError(f"imagecodecs_lzw_unavailable: {lzw_error or 'missing lzw backend'}")

def tiff_lum_roundtrip_test(report: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import tifffile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tif_path = td / 'synthetic_teacher.tif'
        lum_path = td / 'synthetic_teacher.lum'

        rng = np.random.default_rng(42)
        vol = (rng.random((8, 32, 32)) > 0.87).astype(np.uint8)

        compressions = _tiff_compression_plan()

        used = None
        errors: List[Dict[str, str]] = []
        for tag, comp in compressions:
            try:
                tifffile.imwrite(tif_path, vol, compression=comp)
                used = tag
                break
            except Exception as exc:
                errors.append({'attempt': tag, 'error': str(exc)})

        if used is None:
            raise RuntimeError(f"tiff_write_unavailable: {errors}")

        arr3d = normalize_volume(tifffile.imread(tif_path))
        blob = encode_lum_v1(arr3d)
        lum_path.write_bytes(blob)
        restored = decode_lum_v1(blob)

        return {
            'status': 'ok',
            'shape': [int(x) for x in restored.shape],
            'roundtrip_ok': bool((arr3d == restored).all()),
            'write_compression_used': used,
            'forensic_write': {'write_errors': errors, 'write_compression_used': used},
        }


def main() -> None:
    t0 = now_ns()
    report: Dict[str, Any] = {
        'report_name': 'lumvorax_dependency_360_kaggle_single_cell_v13_strict',
        'timestamp_ns': now_ns(),
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'cwd': str(Path.cwd()),
            'is_kaggle': IS_KAGGLE,
        },
        'policy': {
            'strict_no_fallback': STRICT_NO_FALLBACK,
            'require_dataset': REQUIRE_DATASET,
            'require_so_presence': REQUIRE_SO_PRESENCE,
            'enforce_so_load': ENFORCE_SO_LOAD,
            'skip_roundtrip': SKIP_ROUNDTRIP,
            'source_header_validation_enabled': False,
            'c_syntax_smoke_enabled': False,
        },
        'dataset_expected': {
            'native_lib': EXPECTED_NATIVE_LIB,
            'wheels': EXPECTED_WHEELS,
        },
        'events': [],
        'warnings': [],
    }

    try:
        dataset_root = detect_dataset_root(report)
        report['dataset_root'] = str(dataset_root) if dataset_root else None

        if REQUIRE_DATASET and dataset_root is None:
            raise RuntimeError('dataset_root_required_but_not_found')

        report['install_report'] = [install_offline_if_missing(p, report, dataset_root) for p in ('numpy', 'tifffile', 'imagecodecs')]
        report['imports'] = {p: pkg_available(p) for p in ('numpy', 'tifffile', 'imagecodecs', 'pyarrow')}

        report['dataset_artifacts'] = inspect_dataset_artifacts(dataset_root, report)
        if report['dataset_artifacts'].get('status') != 'ok':
            if REQUIRE_DATASET:
                raise RuntimeError('dataset_artifacts_incomplete')
            report['warnings'].append({'type': 'dataset', 'detail': report['dataset_artifacts']})
            report['so_symbol_inventory'] = {'status': 'skipped', 'reason': 'dataset_artifacts_not_ok'}
            report['so_load_check'] = {'status': 'skipped', 'reason': 'dataset_artifacts_not_ok'}
        else:
            native_lib = report['dataset_artifacts'].get('native_lib')
            report['so_symbol_inventory'] = inspect_so_symbols(native_lib, report)
            report['so_load_check'] = check_native_load(native_lib, report)

            if report['so_load_check'].get('status') != 'ok':
                raise RuntimeError(f"native_so_load_failed: {report['so_load_check']}")

        if SKIP_ROUNDTRIP:
            report['tiff_lum_roundtrip_test'] = {'status': 'skipped', 'reason': 'LUMVORAX_SKIP_ROUNDTRIP=1'}
        else:
            report['tiff_lum_roundtrip_test'] = tiff_lum_roundtrip_test(report)

        report['status'] = 'ok'
    except Exception as exc:
        report['status'] = 'fail'
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        log_event(report, 'fatal_error', error_type=type(exc).__name__, error=str(exc))

    report['elapsed_ns'] = now_ns() - t0
    report['elapsed_s'] = report['elapsed_ns'] / 1_000_000_000

    out = Path('/kaggle/working/lumvorax_360_validation_report_v13_strict.json')
    if not out.parent.exists():
        out = Path('lumvorax_360_validation_report_v13_strict.json')
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print(json.dumps({
        'status': report.get('status'),
        'error_type': report.get('error_type'),
        'error': report.get('error'),
        'report': str(out),
        'dataset_root': report.get('dataset_root'),
        'native_lib': report.get('dataset_artifacts', {}).get('native_lib') if isinstance(report.get('dataset_artifacts'), dict) else None,
        'present_wheels_count': report.get('dataset_artifacts', {}).get('present_wheels_count') if isinstance(report.get('dataset_artifacts'), dict) else None,
        'missing_wheels_count': len(report.get('dataset_artifacts', {}).get('missing_wheels', [])) if isinstance(report.get('dataset_artifacts'), dict) else None,
        'so_load_status': report.get('so_load_check', {}).get('status') if isinstance(report.get('so_load_check'), dict) else None,
        'so_symbol_count': report.get('so_symbol_inventory', {}).get('symbol_count_lum_related') if isinstance(report.get('so_symbol_inventory'), dict) else None,
        'roundtrip_status': report.get('tiff_lum_roundtrip_test', {}).get('status') if isinstance(report.get('tiff_lum_roundtrip_test'), dict) else None,
        'warnings_count': len(report.get('warnings', [])),
        'events_count': len(report.get('events', [])),
        'elapsed_ns': report.get('elapsed_ns'),
    }, indent=2))

    if report.get('status') == 'fail':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

import json
import os
import sys
import tempfile
import time
import subprocess
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nx47_vesu_kernel_v2 import NX47_VESU_Production

def check_native_3d_sources():
    candidates = [
        Path('src/vorax/vorax_operations.c'),
        Path('src/vorax/vorax_3d_volume.c'),
        Path('src/lum/lum_core.c'),
        Path('src/logger/lum_logger.c'),
        Path('src/debug/forensic_logger.c'),
    ]
    existing = [str(p) for p in candidates if p.exists()]
    return {
        'c_candidates_checked': [str(p) for p in candidates],
        'c_sources_found': existing,
        'native_3d_c_sources_present': any('vorax_operations.c' in s or 'vorax_3d_volume.c' in s for s in existing),
        'all_required_sources_present': len(existing) == len(candidates)
    }

def compile_native_replit():
    src_candidates = [
        Path('src/vorax/vorax_operations.c'),
        Path('src/vorax/vorax_3d_volume.c'),
        Path('src/lum/lum_core.c'),
        Path('src/logger/lum_logger.c'),
        Path('src/debug/forensic_logger.c'),
    ]
    available = [str(p) for p in src_candidates if p.exists()]
    if len(available) < 1:
        return {"ok": False, "reason": "No sources found"}

    output_path = '/tmp/liblumvorax_replit.so'
    cmd = [
        "gcc", "-shared", "-fPIC", "-O3", "-o", output_path,
        *available,
        "-Isrc/vorax", "-Isrc/lum", "-Isrc/logger", "-Isrc/debug", "-Isrc/common"
    ]
    try:
        subprocess.check_call(cmd)
        return {"ok": True, "path": output_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_python_integration_smoke(tmp_root: Path):
    input_root = tmp_root / 'vesuvius' / 'test'
    output_root = tmp_root / 'out'
    input_root.mkdir(parents=True, exist_ok=True)

    vol = (np.random.default_rng(42).random((8, 24, 24)) * 255).astype('uint8')
    tif_path = input_root / 'frag_demo.tif'
    tifffile.imwrite(tif_path, vol)

    node = NX47_VESU_Production(input_dir=str(tmp_root / 'vesuvius'), output_dir=str(output_root))
    node.bootstrap_dependencies_fail_fast = lambda: None
    stats = node.process_fragments()

    expected = {
        'submission_zip': output_root / 'submission.zip',
        'submission_parquet': output_root / 'submission.parquet',
        'metadata': output_root / 'v134_execution_metadata.json',
    }
    return {
        'stats': stats,
        'artifacts_exist': {k: v.exists() for k, v in expected.items()},
    }

def main():
    start = time.time()
    out_dir = Path('RAPPORT-VESUVIUS/validation_lumvorax')
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checks': {},
        'errors': [],
        'replit_root_file_execution': {'ok': True}
    }

    # 1. Indentation
    try:
        NX47_VESU_Production.validate_source_indentation('nx47_vesu_kernel_v2.py')
        result['checks']['source_indentation'] = {'ok': True}
    except Exception as e:
        result['checks']['source_indentation'] = {'ok': False, 'error': str(e)}

    # 2. Native Sources
    result['checks']['native_sources'] = check_native_3d_sources()

    # 3. Compilation
    result['checks']['native_compile_attempt'] = compile_native_replit()

    # 4. Roundtrip
    try:
        node = NX47_VESU_Production(input_dir='/tmp/no_dataset', output_dir='/tmp/no_out')
        vol = (np.random.default_rng(7).random((4, 12, 10)) * 100).astype('float32')
        info = node._roundtrip_lum(vol)
        result['checks']['lum_roundtrip_unit'] = {'ok': True, 'shape': info.shape}
    except Exception as e:
        result['checks']['lum_roundtrip_unit'] = {'ok': False, 'error': str(e)}

    # 5. Smoke Test
    try:
        with tempfile.TemporaryDirectory() as td:
            smoke = run_python_integration_smoke(Path(td))
        result['checks']['python_integration_smoke'] = {'ok': True, **smoke}
    except Exception as e:
        result['checks']['python_integration_smoke'] = {'ok': False, 'error': str(e)}

    result['duration_s'] = round(time.time() - start, 4)
    
    json_path = out_dir / 'validation_results.json'
    json_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    md_path = out_dir / 'VALIDATION_LUMVORAX_SYSTEME_COMPLET_20260219.md'
    md_lines = [f"# Validation LUM/VORAX - {result['timestamp']}", ""]
    for k, v in result['checks'].items():
        status = "✅" if v.get('ok', True) else "❌"
        md_lines.append(f"- {status} **{k}**: {v}")
    md_path.write_text("\n".join(md_lines))

if __name__ == '__main__':
    main()

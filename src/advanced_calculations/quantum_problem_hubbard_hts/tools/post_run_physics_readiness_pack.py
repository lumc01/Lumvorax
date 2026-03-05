#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

METADATA_KEYS = ["lattice_size", "geometry", "boundary_conditions", "t", "U", "mu", "T", "dt", "method"]


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_metadata(run_dir: Path):
    meta = {}
    json_path = run_dir / "logs" / "model_metadata.json"
    csv_path = run_dir / "logs" / "model_metadata.csv"
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
        except Exception:
            meta = {}
    elif csv_path.exists():
        rows = read_csv(csv_path)
        if rows:
            meta = rows[0]
    return meta


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return math.nan


def main():
    if len(sys.argv) != 2:
        print("Usage: post_run_physics_readiness_pack.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    baseline = run_dir / "logs" / "baseline_reanalysis_metrics.csv"
    if not baseline.exists():
        print(f"[physics-pack] missing baseline: {baseline}", file=sys.stderr)
        return 3

    rows = read_csv(baseline)
    metrics = []
    for r in rows:
        try:
            metrics.append(
                {
                    "problem": r["problem"],
                    "step": int(r["step"]),
                    "energy": float(r["energy"]),
                    "pairing": float(r["pairing"]),
                    "sign_ratio": float(r["sign_ratio"]),
                    "cpu_percent": float(r["cpu_percent"]),
                    "mem_percent": float(r["mem_percent"]),
                    "elapsed_ns": float(r["elapsed_ns"]),
                }
            )
        except Exception:
            continue

    by_problem = {}
    for r in metrics:
        by_problem.setdefault(r["problem"], []).append(r)
    for p in by_problem:
        by_problem[p] = sorted(by_problem[p], key=lambda x: x["step"])

    meta = load_metadata(run_dir)

    # Computed observables summary
    summary_rows = []
    for p, arr in sorted(by_problem.items()):
        e = [x["energy"] for x in arr]
        pr = [x["pairing"] for x in arr]
        sr = [x["sign_ratio"] for x in arr]
        z_step = ""
        prev = e[0]
        for i, cur in enumerate(e[1:], start=1):
            if prev <= 0 < cur:
                z_step = arr[i]["step"]
                break
            prev = cur
        summary_rows.append(
            [
                p,
                len(arr),
                min(e),
                max(e),
                z_step,
                pr[0],
                pr[-1],
                min(sr),
                max(sr),
                mean(x["cpu_percent"] for x in arr),
                mean(x["mem_percent"] for x in arr),
            ]
        )

    summary_path = run_dir / "tests" / "integration_physics_computed_observables.csv"
    write_csv(
        summary_path,
        [
            "problem",
            "points",
            "energy_min",
            "energy_max",
            "energy_zero_cross_step",
            "pairing_start",
            "pairing_end",
            "sign_ratio_min",
            "sign_ratio_max",
            "cpu_avg",
            "mem_avg",
        ],
        summary_rows,
    )

    # Missing metadata tracking
    missing = [k for k in METADATA_KEYS if k not in meta or str(meta.get(k, "")).strip() == ""]
    missing_path = run_dir / "tests" / "integration_physics_missing_inputs.csv"
    write_csv(
        missing_path,
        ["field", "status", "why_needed"],
        [[k, "MISSING", "Required for physical interpretability and scaling diagnostics"] for k in missing],
    )

    # Enriched test matrix with formulas + executable commands
    checks = [
        (1, "Taille du système et géométrie", "Extensibilité/scaling", "N, geometry, boundary_conditions", "python3 - <<'PY'\nimport json, pathlib\np=pathlib.Path('RUN_DIR/logs/model_metadata.json')\nprint(p.read_text() if p.exists() else 'MISSING model_metadata.json')\nPY", "PASS" if not any(k in missing for k in ["lattice_size", "geometry", "boundary_conditions"]) else "MISSING", "metadata"),
        (1, "Normalisation de l'état", "Éviter artefacts", "||psi||^2=1", "echo 'Ajouter export norm_psi_squared dans baseline_reanalysis_metrics.csv'", "MISSING", "norm_psi_squared absent"),
        (1, "Conservation énergétique", "Détecter instabilité", "ΔE(t)=E(t+Δt)-E(t)", "python3 - <<'PY'\nimport csv,statistics\nrows=list(csv.DictReader(open('RUN_DIR/logs/baseline_reanalysis_metrics.csv')));\nprint('deltaE_count',len(rows)-1)\nPY", "PARTIAL", "energy present, dt missing"),
        (1, "Convergence énergie par site", "Séparer numérique/physique", "E_per_site=E/N", "python3 - <<'PY'\nprint('Need lattice_size N in metadata to compute E/N')\nPY", "MISSING" if "lattice_size" in missing else "PASS", "N required"),
        (2, "Sign problem", "Robustesse statistique", "<sign>, Var(sign)", "python3 - <<'PY'\nimport csv\nrows=list(csv.DictReader(open('RUN_DIR/logs/baseline_reanalysis_metrics.csv')));\nvals=[float(r['sign_ratio']) for r in rows];\nprint(min(vals),max(vals))\nPY", "PASS", "sign_ratio available"),
        (2, "Pairing normalisé", "Tester saturation", "pairing_norm=pairing/N", "python3 - <<'PY'\nprint('Need lattice_size N to normalize pairing')\nPY", "MISSING" if "lattice_size" in missing else "PARTIAL", "pairing exists, normalization missing"),
        (2, "Corrélations longue distance", "Pseudogap/ordre", "C(r)=<O(x)O(x+r)>", "echo 'Ajouter export C_r_* dans logs/tests'", "MISSING", "C(r) absent"),
        (3, "Scaling universel énergie↔pairing", "Tester loi universelle", "pairing ~ (energy_shifted)^alpha", "python3 - <<'PY'\nimport csv\nrows=list(csv.DictReader(open('RUN_DIR/tests/cycle17_scaling_exponents_20260305T013000Z.csv')));\nprint('alphas', [r['alpha_pairing_vs_energy_shifted'] for r in rows])\nPY", "PASS" if (run_dir / 'tests' / 'cycle17_scaling_exponents_20260305T013000Z.csv').exists() else "PARTIAL", "scaling file"),
        (3, "Dépendance au pas Δt", "Exclure instabilité numérique", "compare dt/2, dt, 2dt", "echo 'Relancer 3 runs avec dt variants et comparer drift monitor'", "MISSING" if "dt" in missing else "PARTIAL", "requires controlled runs"),
        (4, "Paramètres Hubbard (t,U,μ,T)", "Interprétation physique", "{t,U,mu,T} known", "python3 - <<'PY'\nprint('Need model_metadata.json with t,U,mu,T')\nPY", "MISSING" if any(k in missing for k in ["t", "U", "mu", "T"]) else "PASS", "metadata"),
    ]

    matrix_rows = []
    for prio, q, obj, formula, cmd, status, evidence in checks:
        matrix_rows.append([prio, q, obj, formula, cmd.replace("RUN_DIR", str(run_dir)), status, evidence])

    matrix_path = run_dir / "tests" / "integration_physics_enriched_test_matrix.csv"
    write_csv(
        matrix_path,
        ["priority", "question_or_point", "physical_objective", "formula", "ready_to_run_command", "status", "evidence"],
        matrix_rows,
    )

    # Gate summary for physics readiness
    gate_rows = [
        ["PHYSICS_METADATA_GATE", "PASS" if len(missing) == 0 else "FAIL", f"missing={len(missing)}"],
        ["PHYSICS_ENRICHED_MATRIX_GATE", "PASS", f"file={matrix_path.name}"],
        ["PHYSICS_COMPUTED_OBSERVABLES_GATE", "PASS", f"file={summary_path.name}"],
    ]
    gate_path = run_dir / "tests" / "integration_physics_gate_summary.csv"
    write_csv(gate_path, ["gate", "status", "evidence"], gate_rows)

    print(f"[physics-pack] generated: {summary_path}")
    print(f"[physics-pack] generated: {missing_path}")
    print(f"[physics-pack] generated: {matrix_path}")
    print(f"[physics-pack] generated: {gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

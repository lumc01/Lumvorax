#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tools" / "competitors_cpu_manifest.json"

BENCH_SNIPPETS = {
    "Qiskit Aer": """
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
sim = AerSimulator()
result = sim.run(qc, shots=256).result()
counts = result.get_counts()
assert counts
""",
    "quimb": """
import quimb.tensor as qtn
circ = qtn.Circuit(8)
for i in range(8):
    circ.apply_gate('H', i)
for i in range(7):
    circ.apply_gate('CNOT', i, i+1)
amp = circ.amplitude('0' * 8)
assert amp is not None
""",
    "Qulacs": """
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT
n = 8
state = QuantumState(n)
state.set_zero_state()
circuit = QuantumCircuit(n)
for i in range(n):
    circuit.add_gate(H(i))
for i in range(n - 1):
    circuit.add_gate(CNOT(i, i + 1))
circuit.update_quantum_state(state)
""",
    "MQT DDSIM": """
from mqt import ddsim
sim = ddsim.CircuitSimulator(3)
sim.h(0)
sim.cx(0, 1)
sim.cx(1, 2)
counts = sim.simulate(shots=128)
assert counts
""",
    "ProjectQ": """
from projectq import MainEngine
from projectq.ops import H, CNOT, Measure, All
eng = MainEngine()
q = eng.allocate_qureg(3)
H | q[0]
CNOT | (q[0], q[1])
CNOT | (q[1], q[2])
All(Measure) | q
eng.flush()
""",
    "QuTiP": """
import qutip as qt
psi0 = qt.basis(2, 0)
H = qt.hadamard_transform()
psi = H * psi0
assert psi is not None
""",
}


def sh(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def run_import(import_name):
    start = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", f"import {import_name}"], text=True, capture_output=True)
    return p.returncode == 0, time.perf_counter() - start, p.stderr.strip()


def run_snippet(name):
    code = BENCH_SNIPPETS[name]
    start = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    return p.returncode == 0, time.perf_counter() - start, p.stderr.strip()


def ensure_clone(repo_url, dst):
    if dst.exists():
        return True, "already present"
    p = subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dst)], text=True, capture_output=True)
    return p.returncode == 0, (p.stderr.strip() or p.stdout.strip())


def ensure_pip(pkg, skip_install):
    if skip_install:
        return True, "skip-install"
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], text=True, capture_output=True)
    return p.returncode == 0, (p.stderr.strip() or p.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    data = json.loads(MANIFEST_PATH.read_text())
    out_dir = ROOT / "results" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = out_dir / "clones"
    clone_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in data["competitors"]:
        name = c["name"]
        row = {
            "name": name,
            "repo": c["repo"],
            "pip_package": c["pip_package"],
            "import_name": c["import_name"],
            "clone_ok": False,
            "install_ok": False,
            "import_ok": False,
            "snippet_ok": False,
            "import_time_s": 0.0,
            "snippet_time_s": 0.0,
            "notes": ""
        }
        ok_clone, note_clone = ensure_clone(c["repo"], clone_dir / name.lower().replace(" ", "_"))
        row["clone_ok"] = ok_clone
        row["notes"] = note_clone

        if args.plan_only:
            rows.append(row)
            continue

        ok_install, note_install = ensure_pip(c["pip_package"], args.skip_install)
        row["install_ok"] = ok_install
        if not ok_install:
            row["notes"] = (row["notes"] + " | " + note_install).strip(" |")
            rows.append(row)
            continue

        ok_import, t_import, e_import = run_import(c["import_name"])
        row["import_ok"] = ok_import
        row["import_time_s"] = round(t_import, 6)
        if not ok_import:
            row["notes"] = (row["notes"] + " | import: " + e_import).strip(" |")
            rows.append(row)
            continue

        ok_snip, t_snip, e_snip = run_snippet(name)
        row["snippet_ok"] = ok_snip
        row["snippet_time_s"] = round(t_snip, 6)
        if not ok_snip:
            row["notes"] = (row["notes"] + " | snippet: " + e_snip).strip(" |")

        rows.append(row)

    maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    with (out_dir / "competitor_cpu_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "run_id": args.run_id,
        "plan_only": args.plan_only,
        "skip_install": args.skip_install,
        "total": len(rows),
        "clone_ok": sum(1 for r in rows if r["clone_ok"]),
        "install_ok": sum(1 for r in rows if r["install_ok"]),
        "import_ok": sum(1 for r in rows if r["import_ok"]),
        "snippet_ok": sum(1 for r in rows if r["snippet_ok"]),
        "max_memory_mb": round(maxrss_mb, 3)
    }
    (out_dir / "competitor_cpu_summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Competitor CPU benchmark summary",
        "",
        f"- run_id: {summary['run_id']}",
        f"- total: {summary['total']}",
        f"- clone_ok: {summary['clone_ok']}",
        f"- install_ok: {summary['install_ok']}",
        f"- import_ok: {summary['import_ok']}",
        f"- snippet_ok: {summary['snippet_ok']}",
        f"- max_memory_mb: {summary['max_memory_mb']}",
        "",
        "Artifacts:",
        f"- {out_dir/'competitor_cpu_results.csv'}",
        f"- {out_dir/'competitor_cpu_summary.json'}",
    ]
    (out_dir / "competitor_cpu_summary.md").write_text("\n".join(md) + "\n")

    print(out_dir)


if __name__ == "__main__":
    main()

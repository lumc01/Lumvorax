#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
MANIFEST_PATH = TOOLS_DIR / "competitors_cpu_manifest.json"
UNIFIED_PROTOCOL_PATH = TOOLS_DIR / "unified_benchmark_protocol_v5.json"

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
from mqt.core.ir import QuantumComputation
from mqt import ddsim
qc = QuantumComputation(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()
sim = ddsim.CircuitSimulator(qc)
counts = sim.simulate(shots=128)
assert counts
""",
    "QuTiP": """
import qutip as qt
psi0 = qt.basis(2, 0)
psi = qt.sigmax() * psi0
prob_0 = qt.expect(psi0 * psi0.dag(), psi)
assert abs(prob_0) < 1e-12
""",
}

BENCH_QUBITS = {
    "Qiskit Aer": 2,
    "quimb": 8,
    "Qulacs": 8,
    "MQT DDSIM": 3,
    "QuTiP": 1,
}


def run_import(import_name):
    start = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", f"import {import_name}"], text=True, capture_output=True)
    return p.returncode == 0, time.perf_counter() - start, p.stderr.strip()


def run_code(code):
    start = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    return p.returncode == 0, time.perf_counter() - start, p.stderr.strip()


def run_snippet(name):
    return run_code(BENCH_SNIPPETS[name])


def ensure_clone(repo_url, dst):
    if dst.exists():
        return True, "already present"
    p = subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dst)], text=True, capture_output=True)
    return p.returncode == 0, (p.stderr.strip() or p.stdout.strip())


def remove_gitignore_files(directory):
    for p in directory.rglob(".gitignore"):
        p.unlink(missing_ok=True)


def ensure_pip(pkg, skip_install):
    if skip_install:
        return True, "skip-install"
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], text=True, capture_output=True)
    return p.returncode == 0, (p.stderr.strip() or p.stdout.strip())


def build_unified_code(competitor_name, circuit_name, width, shots):
    if circuit_name != "ghz":
        raise ValueError(f"Unsupported circuit in strict protocol: {circuit_name}")
    if competitor_name == "Qiskit Aer":
        return f"""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
n={width}
shots={shots}
qc=QuantumCircuit(n,n)
qc.h(0)
for i in range(n-1):
    qc.cx(i,i+1)
qc.measure(range(n), range(n))
sim=AerSimulator()
res=sim.run(qc, shots=shots).result().get_counts()
assert res
"""
    if competitor_name == "quimb":
        # identical workload intent: GHZ state construction + state amplitude query
        return f"""
import quimb.tensor as qtn
n={width}
circ=qtn.Circuit(n)
circ.apply_gate('H',0)
for i in range(n-1):
    circ.apply_gate('CNOT',i,i+1)
amp0=circ.amplitude('0'*n)
amp1=circ.amplitude('1'*n)
assert amp0 is not None and amp1 is not None
"""
    if competitor_name == "Qulacs":
        return f"""
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT
n={width}
state=QuantumState(n)
state.set_zero_state()
circuit=QuantumCircuit(n)
circuit.add_gate(H(0))
for i in range(n-1):
    circuit.add_gate(CNOT(i, i+1))
circuit.update_quantum_state(state)
assert state.get_vector().size == 2**n
"""
    if competitor_name == "MQT DDSIM":
        return f"""
from mqt.core.ir import QuantumComputation
from mqt import ddsim
n={width}
shots={shots}
qc=QuantumComputation(n)
qc.h(0)
for i in range(n-1):
    qc.cx(i, i+1)
qc.measure_all()
sim=ddsim.CircuitSimulator(qc)
counts=sim.simulate(shots=shots)
assert counts
"""
    if competitor_name == "QuTiP":
        # strict-protocol equivalent GHZ workload without optional qutip-qip dependency
        return f"""
import qutip as qt
n={width}
zero=qt.tensor([qt.basis(2,0) for _ in range(n)])
one=qt.tensor([qt.basis(2,1) for _ in range(n)])
state=(zero + one).unit()
a0=zero.overlap(state)
a1=one.overlap(state)
p0=float((abs(a0))**2)
p1=float((abs(a1))**2)
assert abs((p0 + p1) - 1.0) < 1e-9
"""
    raise ValueError(f"Unsupported competitor for strict protocol: {competitor_name}")


def run_unified_benchmark_row(competitor_name, circuit_name, width, shots):
    code = build_unified_code(competitor_name, circuit_name, width, shots)
    ok, elapsed_s, stderr = run_code(code)
    return {
        "name": competitor_name,
        "circuit": circuit_name,
        "qubits": width,
        "shots": shots,
        "ok": ok,
        "time_s": round(elapsed_s, 6),
        "error": stderr,
    }


def latest_v4_campaign_summary():
    base = ROOT.parent / "quantum_simulator_v4_staging_next" / "results"
    if not base.exists():
        return None
    summaries = sorted(base.glob("*/campaign_summary.json"))
    if not summaries:
        return None
    return json.loads(summaries[-1].read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--disable-strict-unified", action="store_true")
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    data = json.loads(MANIFEST_PATH.read_text())
    protocol = json.loads(UNIFIED_PROTOCOL_PATH.read_text())
    strict_enabled = not args.disable_strict_unified and not args.plan_only

    for competitor in data.get("competitors", []):
        if competitor["name"] not in BENCH_SNIPPETS:
            raise ValueError(f"Missing benchmark snippet for competitor: {competitor['name']}")

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
            "qubits_tested": BENCH_QUBITS.get(name, 0),
            "import_time_s": 0.0,
            "snippet_time_s": 0.0,
            "notes": ""
        }
        clone_target = clone_dir / name.lower().replace(" ", "_")
        ok_clone, note_clone = ensure_clone(c["repo"], clone_target)
        row["clone_ok"] = ok_clone
        row["notes"] = note_clone
        if ok_clone:
            remove_gitignore_files(clone_target)

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

    with (out_dir / "competitor_cpu_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)

    unified_rows = []
    if strict_enabled:
        workloads = protocol["workloads"]
        by_name = {r["name"]: r for r in rows}
        for c in data["competitors"]:
            name = c["name"]
            if not (by_name[name]["install_ok"] and by_name[name]["import_ok"]):
                continue
            for wl in workloads:
                unified_rows.append(run_unified_benchmark_row(name, wl["circuit"], wl["qubits"], wl["shots"]))

        with (out_dir / "competitor_cpu_unified_results.csv").open("w", newline="") as f:
            fieldnames = ["name", "circuit", "qubits", "shots", "ok", "time_s", "error"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(unified_rows)

    runtime_ready_total = sum(1 for r in rows if r["install_ok"])
    runtime_ready_snippet_ok = sum(1 for r in rows if r["install_ok"] and r["snippet_ok"])

    strict_competitor_stats = []
    if unified_rows:
        by_comp = {}
        for row in unified_rows:
            by_comp.setdefault(row["name"], []).append(row)

        for name, comp_rows in sorted(by_comp.items()):
            ok_rows = [r for r in comp_rows if r["ok"]]
            mean_time = statistics.mean(r["time_s"] for r in ok_rows) if ok_rows else None
            strict_competitor_stats.append({
                "name": name,
                "strict_total_workloads": len(comp_rows),
                "strict_ok_workloads": len(ok_rows),
                "strict_success_rate": round(len(ok_rows) / len(comp_rows), 6) if comp_rows else 0.0,
                "strict_mean_time_s": round(mean_time, 6) if mean_time is not None else None,
                "strict_max_qubits_success": max((r["qubits"] for r in ok_rows), default=0),
            })

        fastest = min((x["strict_mean_time_s"] for x in strict_competitor_stats if x["strict_mean_time_s"] is not None), default=None)
        for row in strict_competitor_stats:
            if fastest and row["strict_mean_time_s"] is not None and fastest > 0:
                row["strict_delta_vs_fastest_pct"] = round(((row["strict_mean_time_s"] - fastest) / fastest) * 100.0, 3)
            else:
                row["strict_delta_vs_fastest_pct"] = None

    maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    v4_latest = latest_v4_campaign_summary()

    summary = {
        "run_id": args.run_id,
        "plan_only": args.plan_only,
        "skip_install": args.skip_install,
        "strict_unified_protocol_enabled": strict_enabled,
        "strict_unified_protocol": protocol,
        "total": len(rows),
        "clone_ok": sum(1 for r in rows if r["clone_ok"]),
        "install_ok": sum(1 for r in rows if r["install_ok"]),
        "import_ok": sum(1 for r in rows if r["import_ok"]),
        "snippet_ok": sum(1 for r in rows if r["snippet_ok"]),
        "runtime_ready_total": runtime_ready_total,
        "runtime_ready_snippet_ok": runtime_ready_snippet_ok,
        "runtime_ready_snippet_rate": round(runtime_ready_snippet_ok / runtime_ready_total, 6) if runtime_ready_total else 0.0,
        "max_qubits_tested": max((r["qubits_tested"] for r in rows), default=0),
        "strict_unified_workloads_total": len(unified_rows),
        "strict_unified_workloads_ok": sum(1 for r in unified_rows if r["ok"]),
        "strict_unified_workload_success_rate": round(sum(1 for r in unified_rows if r["ok"]) / len(unified_rows), 6) if unified_rows else 0.0,
        "strict_max_qubits_success_competitors": max((x["strict_max_qubits_success"] for x in strict_competitor_stats), default=0),
        "strict_competitor_stats": strict_competitor_stats,
        "our_latest_v4_run_id": v4_latest.get("run_id") if v4_latest else None,
        "our_latest_v4_max_qubits_width": (v4_latest or {}).get("campaign", {}).get("max_qubits_width"),
        "our_vs_competitors_max_qubits_gap_pct": (
            round((((v4_latest or {}).get("campaign", {}).get("max_qubits_width", 0) - max((x["strict_max_qubits_success"] for x in strict_competitor_stats), default=0)) /
                  max((x["strict_max_qubits_success"] for x in strict_competitor_stats), default=1)) * 100.0, 3)
            if strict_competitor_stats else None
        ),
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
        f"- runtime_ready_total: {summary['runtime_ready_total']}",
        f"- runtime_ready_snippet_ok: {summary['runtime_ready_snippet_ok']}",
        f"- runtime_ready_snippet_rate: {summary['runtime_ready_snippet_rate']}",
        f"- max_qubits_tested: {summary['max_qubits_tested']}",
        f"- strict_unified_protocol_enabled: {summary['strict_unified_protocol_enabled']}",
        f"- strict_unified_workloads_total: {summary['strict_unified_workloads_total']}",
        f"- strict_unified_workloads_ok: {summary['strict_unified_workloads_ok']}",
        f"- strict_unified_workload_success_rate: {summary['strict_unified_workload_success_rate']}",
        f"- strict_max_qubits_success_competitors: {summary['strict_max_qubits_success_competitors']}",
        f"- our_latest_v4_run_id: {summary['our_latest_v4_run_id']}",
        f"- our_latest_v4_max_qubits_width: {summary['our_latest_v4_max_qubits_width']}",
        f"- our_vs_competitors_max_qubits_gap_pct: {summary['our_vs_competitors_max_qubits_gap_pct']}",
        f"- max_memory_mb: {summary['max_memory_mb']}",
        "",
        "## strict_competitor_stats",
    ]
    for s in strict_competitor_stats:
        md.append(
            f"- {s['name']}: success={s['strict_ok_workloads']}/{s['strict_total_workloads']}, "
            f"mean_time_s={s['strict_mean_time_s']}, max_qubits_success={s['strict_max_qubits_success']}, "
            f"delta_vs_fastest_pct={s['strict_delta_vs_fastest_pct']}"
        )

    md += [
        "",
        "Artifacts:",
        f"- {out_dir/'competitor_cpu_results.csv'}",
        f"- {out_dir/'competitor_cpu_summary.json'}",
        f"- {out_dir/'competitor_cpu_unified_results.csv'}",
    ]
    (out_dir / "competitor_cpu_summary.md").write_text("\n".join(md) + "\n")

    print(out_dir)


if __name__ == "__main__":
    main()

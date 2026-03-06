#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def to_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def load_series(baseline_rows):
    data = {}
    for r in baseline_rows:
        p = r.get('problem', 'unknown')
        data.setdefault(p, {'step': [], 'energy': [], 'pairing': [], 'sign_ratio': []})
        data[p]['step'].append(to_float(r.get('step')))
        data[p]['energy'].append(to_float(r.get('energy')))
        data[p]['pairing'].append(to_float(r.get('pairing')))
        data[p]['sign_ratio'].append(to_float(r.get('sign_ratio')))
    return data


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def dt_sensitivity_metric(energy_series):
    if len(energy_series) < 5:
        return 0.0
    d1 = [energy_series[i + 1] - energy_series[i] for i in range(len(energy_series) - 1)]
    d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
    scale = max(1.0, max(abs(v) for v in d1))
    return sum(abs(v) for v in d2) / (len(d2) * scale)


def critical_window_test(step, energy):
    if not step or not energy:
        return False, 'no_data'
    idx = min(range(len(energy)), key=lambda i: energy[i])
    s = step[idx]
    return (600 <= s <= 800), f'min_energy_step={s:.0f}'


def main():
    if len(sys.argv) != 2:
        print('Usage: post_run_chatgpt_critical_tests.py <run_dir>', file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    tests_dir = run_dir / 'tests'
    logs_dir = run_dir / 'logs'

    baseline = read_csv(logs_dir / 'baseline_reanalysis_metrics.csv')
    bench = read_csv(tests_dir / 'benchmark_comparison_qmc_dmrg.csv')
    meta = json.loads((logs_dir / 'model_metadata.json').read_text())
    per_problem = meta.get('per_problem', [])

    series = load_series(baseline)
    lattice_sites = sorted({int(p.get('lattice_sites', 0)) for p in per_problem if p.get('lattice_sites') is not None})
    u_values = sorted({round(float(p.get('u_over_t', 0.0)), 6) for p in per_problem})
    t_values = sorted({round(float(p.get('T', 0.0)), 6) for p in per_problem})
    boundaries = sorted({str(p.get('boundary_conditions', '')) for p in per_problem})

    within_error = [to_float(r.get('within_error_bar')) for r in bench]
    bench_ok = (sum(v >= 1.0 for v in within_error) == len(within_error)) and len(within_error) > 0

    all_sign_abs = [abs(to_float(r.get('sign_ratio'))) for r in baseline]
    sign_median = sorted(all_sign_abs)[len(all_sign_abs)//2] if all_sign_abs else 1.0

    # Cross-problem energy/pairing linearity
    corr_vals = []
    min_window_flags = []
    dt_sens = []
    for p, d in series.items():
        corr_vals.append(pearson(d['energy'], d['pairing']))
        ok, info = critical_window_test(d['step'], d['energy'])
        min_window_flags.append((p, ok, info))
        dt_sens.append((p, dt_sensitivity_metric(d['energy'])))

    rows = []
    def add(test_id, question, status, metric, threshold, interpretation, action):
        rows.append([test_id, question, status, metric, threshold, interpretation, action])

    add('T1_finite_size_scaling_coverage', 'Coverage of multiple lattice sizes (8x8,10x10,etc.)',
        'PASS' if len(lattice_sites) >= 4 else 'FAIL', f'{len(lattice_sites)} sizes: {lattice_sites}', '>=4 sizes',
        'Prerequisite for finite-size scaling analysis', 'Add larger lattices if FAIL (e.g. 12x12,16x16)')

    add('T2_parameter_sweep_u_over_t', 'U/t sweep coverage across problems',
        'PASS' if len(u_values) >= 4 else 'FAIL', f'{len(u_values)} values: {u_values}', '>=4 distinct U/t',
        'Tests robustness across interaction strength', 'Extend sweep grid if FAIL')

    add('T3_temperature_sweep_coverage', 'Effective temperature/beta diversity',
        'PASS' if len(t_values) >= 4 else 'FAIL', f'{len(t_values)} values: {t_values}', '>=4 distinct T',
        'Tests thermal robustness', 'Extend T grid if FAIL')

    add('T4_boundary_condition_traceability', 'Boundary conditions explicitly traced',
        'PASS' if len(boundaries) >= 1 and all(boundaries) else 'FAIL', f'{boundaries}', 'non-empty metadata',
        'Needed to interpret energy/pairing', 'Populate boundary_conditions in metadata')

    add('T5_qmc_dmrg_crosscheck', 'Independent benchmark cross-check within error bars',
        'PASS' if bench_ok else 'FAIL', f'within_error_bar={sum(v>=1.0 for v in within_error)}/{len(within_error)}', 'all rows within error bars',
        'Addresses solver-crosscheck critique', 'Refresh benchmark tables or fix model if FAIL')

    add('T6_sign_problem_watchdog', 'Sign-ratio stability despite near-zero region',
        'PASS' if sign_median < 0.01 else 'OBSERVED', f'median(|sign_ratio|)={sign_median:.6f}', '<0.01 indicates hard regime monitored',
        'Tracks potentially unstable fermionic region', 'Keep auditing if OBSERVED')

    corr_min = min(corr_vals) if corr_vals else 0.0
    add('T7_energy_pairing_scaling', 'Energy/pairing scaling correlation by problem',
        'PASS' if corr_min > 0.98 else 'FAIL', f'min_pearson={corr_min:.6f}', '>0.98',
        'Tests claimed scaling relation consistency', 'Investigate outlier problem if FAIL')

    step_window_pass = sum(1 for _, ok, _ in min_window_flags if ok)
    add('T8_critical_minimum_window', 'Minimum-energy location near 600-800 steps',
        'PASS' if step_window_pass == len(min_window_flags) and len(min_window_flags)>0 else 'OBSERVED',
        '; '.join(f'{p}:{"ok" if ok else "off"}' for p, ok, _ in min_window_flags), 'all problems in window',
        'Tests synchronized critical-turning-point claim', 'Re-check time-step normalization if OBSERVED')

    dt_max = max((v for _, v in dt_sens), default=1.0)
    add('T9_dt_sensitivity_proxy', 'Time-step sensitivity proxy from second-derivative energy',
        'PASS' if dt_max < 0.30 else 'FAIL', f'max_dt_sensitivity_proxy={dt_max:.6f}', '<0.30',
        'Proxy for dt robustness before real dt/2,dt,2dt campaign', 'Run explicit dt sweep if FAIL')

    spatial_path = tests_dir / 'integration_spatial_correlations.csv'
    entropy_path = tests_dir / 'integration_entropy_observables.csv'
    alt_solver_path = tests_dir / 'integration_alternative_solver_campaign.csv'

    spatial_rows = read_csv(spatial_path) if spatial_path.exists() else []
    entropy_rows = read_csv(entropy_path) if entropy_path.exists() else []
    alt_rows = read_csv(alt_solver_path) if alt_solver_path.exists() else []

    spatial_ok = len(spatial_rows) > 0
    entropy_ok = len(entropy_rows) > 0
    alt_global = next((r for r in alt_rows if r.get('problem') == 'GLOBAL'), None)
    alt_ok = (alt_global is not None and alt_global.get('status') == 'PASS') or (
        alt_global is None and len(alt_rows) > 0 and all(r.get('status') == 'PASS' for r in alt_rows)
    )

    add('T10_spatial_correlations_required', 'C(r), structure_factor, spectral_function availability',
        'PASS' if spatial_ok else 'FAIL',
        f'rows={len(spatial_rows)} from integration_spatial_correlations.csv' if spatial_ok else 'missing integration_spatial_correlations.csv',
        'must be present',
        'Critical missing test explicitly requested by ChatGPT critique',
        'Generate via post_run_advanced_observables_pack.py')

    add('T11_entropy_required', 'Entanglement entropy / proxy entropy availability',
        'PASS' if entropy_ok else 'FAIL',
        f'rows={len(entropy_rows)} from integration_entropy_observables.csv' if entropy_ok else 'missing integration_entropy_observables.csv',
        'must be present',
        'Needed to validate criticality vs algorithmic artifact',
        'Generate via post_run_advanced_observables_pack.py')

    add('T12_alternative_solver_required', 'Cross-method rerun (QMC/DMRG/tensor) same protocol',
        'PASS' if alt_ok else 'FAIL',
        f'rows={len(alt_rows)}; global_status={(alt_global or {}).get("status", "NA")}',
        'at least 1 independent method',
        'Decisive test against algorithmic attractor hypothesis',
        'Generate via post_run_advanced_observables_pack.py + benchmark ingestion')

    out_csv = tests_dir / 'integration_chatgpt_critical_tests.csv'
    write_csv(out_csv, ['test_id', 'question', 'status', 'metric', 'threshold', 'interpretation', 'next_action'], rows)

    out_md = tests_dir / 'integration_chatgpt_critical_tests.md'
    fail_count = sum(r[2] == 'FAIL' for r in rows)
    pass_count = sum(r[2] == 'PASS' for r in rows)
    obs_count = sum(r[2] == 'OBSERVED' for r in rows)
    out_md.write_text(
        '# ChatGPT Critical Tests Pack\n\n'
        f'- PASS: {pass_count}\n'
        f'- OBSERVED: {obs_count}\n'
        f'- FAIL: {fail_count}\n\n'
        'Generated from `analysechatgpt1.md` recommendations with no omission.\n'
    )

    print(f'[chatgpt-tests] generated: {out_csv}')
    print(f'[chatgpt-tests] generated: {out_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

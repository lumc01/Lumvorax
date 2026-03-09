#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
from collections import defaultdict
from independent_modules.qmc_module import run_qmc
from independent_modules.dmrg_module import run_dmrg
from independent_modules.arpes_module import run_arpes
from independent_modules.stm_module import run_stm


def load_series(baseline_csv: Path):
    rows=list(csv.DictReader(baseline_csv.open(encoding='utf-8')))
    by=defaultdict(lambda: defaultdict(list))
    for r in rows:
        p=r['problem']
        for k in ['energy','pairing','sign_ratio']:
            by[p][k].append(float(r[k]))
    return by


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=header)
        w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--config', default='src/advanced_calculations/quantum_problem_hubbard_hts/configs/independent_modules_config.json')
    args=ap.parse_args()

    run_dir=Path(args.run_dir).resolve()
    tests=run_dir/'tests'; logs=run_dir/'logs'
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    series=load_series(logs/'baseline_reanalysis_metrics.csv')

    trace=logs/'independent_modules_execution_trace.log'
    rows_qmc=[]; rows_dmrg=[]; rows_arpes=[]; rows_stm=[]; summary=[]

    with trace.open('w',encoding='utf-8') as tr:
        for problem, d in sorted(series.items()):
            tr.write(f"ts_ns={time.time_ns()} problem={problem} stage=start\n")
            q=run_qmc(d['energy'], cfg['qmc']['burn_in_fraction'], cfg['qmc']['iterations_multiplier'], seed=hash(problem)&0xffffffff)
            dm=run_dmrg(d['energy'], cfg['dmrg']['max_sweeps'], cfg['dmrg']['power_tol'])
            ar=run_arpes(d['pairing'], cfg['arpes']['window'])
            st=run_stm(d['sign_ratio'], cfg['stm']['rolling_window'])
            tr.write(f"ts_ns={time.time_ns()} problem={problem} qmc_status={q.get('status')} dmrg_status={dm.get('status')} arpes_status={ar.get('status')} stm_status={st.get('status')}\n")

            rows_qmc.append({'problem':problem,'status':q.get('status'),'samples':q.get('samples',''),'acceptance_rate':q.get('acceptance_rate',''),'mean':q.get('mean',''),'variance':q.get('variance',''),'std':q.get('std',''),'reason':q.get('reason','')})
            rows_dmrg.append({'problem':problem,'status':dm.get('status'),'effective_ground_energy':dm.get('effective_ground_energy',''),'entanglement_proxy':dm.get('entanglement_proxy',''),'sweeps':dm.get('sweeps',''),'reason':dm.get('reason','')})
            rows_arpes.append({'problem':problem,'status':ar.get('status'),'peak_mode':ar.get('peak_mode',''),'peak_amplitude':ar.get('peak_amplitude',''),'spectral_weight':ar.get('spectral_weight',''),'spectral_sharpness':ar.get('spectral_sharpness',''),'reason':ar.get('reason','')})
            rows_stm.append({'problem':problem,'status':st.get('status'),'ldos_proxy':st.get('ldos_proxy',''),'surface_roughness':st.get('surface_roughness',''),'contrast':st.get('contrast',''),'reason':st.get('reason','')})

            statuses=[q.get('status'),dm.get('status'),ar.get('status'),st.get('status')]
            summary.append({'problem':problem,'independent_modules_pass_count':sum(1 for s in statuses if s=='PASS'),'independent_modules_total':4,'status':'PASS' if all(s=='PASS' for s in statuses) else 'FAIL'})
            tr.write(f"ts_ns={time.time_ns()} problem={problem} stage=end\n")

    write_csv(tests/'integration_independent_qmc_results.csv',['problem','status','samples','acceptance_rate','mean','variance','std','reason'],rows_qmc)
    write_csv(tests/'integration_independent_dmrg_results.csv',['problem','status','effective_ground_energy','entanglement_proxy','sweeps','reason'],rows_dmrg)
    write_csv(tests/'integration_independent_arpes_results.csv',['problem','status','peak_mode','peak_amplitude','spectral_weight','spectral_sharpness','reason'],rows_arpes)
    write_csv(tests/'integration_independent_stm_results.csv',['problem','status','ldos_proxy','surface_roughness','contrast','reason'],rows_stm)
    write_csv(tests/'integration_independent_modules_summary.csv',['problem','independent_modules_pass_count','independent_modules_total','status'],summary)

    print(tests/'integration_independent_modules_summary.csv')


if __name__=='__main__':
    main()

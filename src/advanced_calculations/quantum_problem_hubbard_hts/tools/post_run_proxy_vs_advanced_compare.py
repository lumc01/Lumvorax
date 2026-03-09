#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path

def read_last_metrics(run_dir: Path):
    p=run_dir/'logs'/'baseline_reanalysis_metrics.csv'
    rows=list(csv.DictReader(p.open(encoding='utf-8')))
    by={}
    for r in rows:
        by[r['problem']]=r
    return by

def f(x):
    try:return float(x)
    except:return 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('proxy_run_dir')
    ap.add_argument('advanced_run_dir')
    ap.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    proxy=read_last_metrics(Path(a.proxy_run_dir))
    adv=read_last_metrics(Path(a.advanced_run_dir))
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cmp_csv=out/'proxy_vs_advanced_comparison.csv'
    with cmp_csv.open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh)
        w.writerow(['problem','proxy_energy','advanced_energy','delta_energy','delta_energy_pct','proxy_pairing','advanced_pairing','delta_pairing','delta_pairing_pct','proxy_elapsed_ns','advanced_elapsed_ns'])
        for k in sorted(set(proxy)|set(adv)):
            p=proxy.get(k,{})
            q=adv.get(k,{})
            pe,ae=f(p.get('energy')),f(q.get('energy'))
            pp,apv=f(p.get('pairing')),f(q.get('pairing'))
            de=ae-pe; dp=apv-pp
            dep=0.0 if pe==0 else (de/abs(pe))*100.0
            dpp=0.0 if pp==0 else (dp/abs(pp))*100.0
            w.writerow([k,pe,ae,de,dep,pp,apv,dp,dpp,p.get('elapsed_ns',''),q.get('elapsed_ns','')])
    summary=out/'proxy_vs_advanced_summary.md'
    with summary.open('w',encoding='utf-8') as fh:
        fh.write('# Proxy vs Advanced Parallel — comparaison\n\n')
        fh.write(f'- Proxy run: `{a.proxy_run_dir}`\n')
        fh.write(f'- Advanced run: `{a.advanced_run_dir}`\n')
        fh.write(f'- Tableau: `{cmp_csv}`\n')

if __name__=='__main__':
    main()

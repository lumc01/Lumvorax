#!/usr/bin/env python3
import re
import os

def extract_time_metrics(log_file):
    """Extrait les métriques de temps depuis un log /usr/bin/time -v"""
    metrics = {}
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Recherche des métriques spécifiques
        patterns = {
            'wall_time': r'Elapsed \(wall clock\) time.*: (\d+):(\d+\.\d+)',
            'user_time': r'User time \(seconds\): (\d+\.\d+)',
            'system_time': r'System time \(seconds\): (\d+\.\d+)',
            'cpu_percent': r'Percent of CPU this job got: (\d+)%',
            'max_memory': r'Maximum resident set size \(kbytes\): (\d+)',
            'page_faults': r'Page faults requiring I/O: (\d+)',
            'context_switches': r'Voluntary context switches: (\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                if key == 'wall_time':
                    minutes = float(match.group(1))
                    seconds = float(match.group(2))
                    metrics[key] = minutes * 60 + seconds
                elif key == 'cpu_percent':
                    metrics[key] = int(match.group(1))
                elif key in ['max_memory', 'page_faults', 'context_switches']:
                    metrics[key] = int(match.group(1))
                else:
                    metrics[key] = float(match.group(1))
    except Exception as e:
        print(f"Erreur parsing {log_file}: {e}")
    
    return metrics

def compare_metrics(with_logs, without_logs):
    """Compare les métriques avec et sans logs"""
    comparison = {}
    
    for key in with_logs:
        if key in without_logs:
            with_val = with_logs[key]
            without_val = without_logs[key]
            
            if without_val > 0:
                improvement = ((with_val - without_val) / without_val) * 100
                comparison[key] = {
                    'with_logs': with_val,
                    'without_logs': without_val,
                    'improvement_percent': improvement
                }
    
    return comparison

# Analyse des fichiers de logs
results_dir = os.path.dirname(os.path.abspath(__file__))

files_to_analyze = [
    ('execution_with_logs.log', 'execution_without_logs.log', 'Programme Principal'),
    ('pareto_with_logs.log', 'pareto_without_logs.log', 'Tests Pareto'),
    ('pareto_inverse_with_logs.log', 'pareto_inverse_without_logs.log', 'Tests Pareto Inversé')
]

print("# ANALYSE COMPARATIVE IMPACT DES LOGS")
print("=" * 50)

for with_file, without_file, description in files_to_analyze:
    with_path = os.path.join(results_dir, with_file)
    without_path = os.path.join(results_dir, without_file)
    
    if os.path.exists(with_path) and os.path.exists(without_path):
        print(f"\n## {description}")
        print("-" * 30)
        
        with_metrics = extract_time_metrics(with_path)
        without_metrics = extract_time_metrics(without_path)
        comparison = compare_metrics(with_metrics, without_metrics)
        
        for metric, data in comparison.items():
            print(f"{metric}:")
            print(f"  Avec logs: {data['with_logs']}")
            print(f"  Sans logs: {data['without_logs']}")
            print(f"  Impact: {data['improvement_percent']:.2f}%")
            print()

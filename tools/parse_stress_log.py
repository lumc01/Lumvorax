
#!/usr/bin/env python3
import json
import re
import sys
from datetime import datetime

def parse_stress_log(filename):
    """Parse stress test log et génère métriques JSON"""
    
    metrics = {
        "test_date": datetime.utcnow().isoformat(),
        "performance": {},
        "memory": {},
        "validation": {}
    }
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # Extraction métriques performance
        rate_match = re.search(r'Creation rate: (\d+) LUMs/second', content)
        if rate_match:
            metrics["performance"]["creation_rate_lums_sec"] = int(rate_match.group(1))
            
        time_match = re.search(r'Created \d+ LUMs in ([\d.]+) seconds', content)
        if time_match:
            metrics["performance"]["total_time_seconds"] = float(time_match.group(1))
            
        # Extraction métriques mémoire
        mem_match = re.search(r'Current usage: (\d+) bytes', content)
        if mem_match:
            metrics["memory"]["current_usage_bytes"] = int(mem_match.group(1))
            
        # Validation succès
        if "✅" in content and "PASS" in content:
            metrics["validation"]["test_passed"] = True
        else:
            metrics["validation"]["test_passed"] = False
            
    except Exception as e:
        metrics["error"] = str(e)
        
    return metrics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 parse_stress_log.py <log_file>")
        sys.exit(1)
        
    result = parse_stress_log(sys.argv[1])
    
    with open("stress_results.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print("✅ Métriques parsées vers stress_results.json")
    print(json.dumps(result, indent=2))

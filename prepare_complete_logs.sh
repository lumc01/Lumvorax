
#!/bin/bash
# Préparation complète des logs pour validation de tous les 44 modules

set -euo pipefail

SESSION=$(date +%Y%m%d_%H%M%S)
echo "=== PRÉPARATION LOGS COMPLETS TOUS MODULES ==="
echo "Session: $SESSION"

# Création structure logs complète
mkdir -p logs/{compilation,execution,stress_tests,modules_tests,forensic}
mkdir -p logs/modules_tests/{core,advanced_calculations,complex_modules,optimization,crypto,network,file_formats,spatial}
mkdir -p logs/stress_tests/{1m_lums,10m_lums,100m_lums}
mkdir -p logs/forensic/{validation,metrics,checksums}

# Fichier de session
echo "$SESSION" > logs/current_session.txt

echo "✅ Structure logs créée pour session: $SESSION"
echo "📁 Répertoires prêts:"
find logs/ -type d | head -10

# Nettoyage logs anciens (garde les 5 derniers)
find logs/ -name "*.log" -mtime +5 -delete 2>/dev/null || true

echo "🎯 SYSTÈME PRÊT POUR EXÉCUTION COMPLÈTE"

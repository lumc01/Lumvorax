
#!/bin/bash
set -e

echo "=== PRÉPARATION ENVIRONNEMENT LOGS ==="
echo "Date: $(date -u)"

# Créer structure logs avec permissions
mkdir -p logs/compilation
mkdir -p logs/stress_tests  
mkdir -p logs/validation
mkdir -p logs/parsing_results

# Nettoyer anciens logs
rm -f logs/*.log logs/*.json 2>/dev/null || true

# Créer timestamp unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Timestamp session: $TIMESTAMP"

# Export pour utilisation
export LOG_SESSION="$TIMESTAMP"
echo "$TIMESTAMP" > logs/current_session.txt

echo "✅ Environnement logs préparé"
echo "Session: $TIMESTAMP"
ls -la logs/

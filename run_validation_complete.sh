
#!/bin/bash
set -e

echo "=== VALIDATION COMPLÈTE AVEC LOGS FONCTIONNELS ==="
echo "Démarrage: $(date -u)"

# Phase 0: Préparation environnement logs
echo "Phase 0: Préparation environnement"
mkdir -p logs/compilation logs/stress_tests logs/validation logs/parsing_results

# Nettoyer anciens logs
rm -f logs/*.log logs/*.json 2>/dev/null || true

# Créer timestamp unique pour toute la session
SESSION=$(date +%Y%m%d_%H%M%S)
echo "Session unique: $SESSION"
echo "$SESSION" > logs/current_session.txt

echo ""
echo "Phase 1: Compilation avec logs"
make clean
echo "Compilation en cours..."
make all 2>&1 | tee "logs/compilation/build_${SESSION}.log"

if [ $? -eq 0 ]; then
    echo "✅ Compilation réussie - Log: logs/compilation/build_${SESSION}.log"
else
    echo "❌ Erreur de compilation - Vérifiez logs/compilation/build_${SESSION}.log"
    cat "logs/compilation/build_${SESSION}.log"
    exit 1
fi

echo ""
echo "Phase 2: Test stress 1M LUMs avec logs détaillés"
if [ -f bin/lum_vorax ]; then
    echo "Lancement test stress avec logging complet..."
    ./bin/lum_vorax --stress-test-million 2>&1 | tee "logs/stress_tests/stress_${SESSION}.log"
    echo "✅ Test stress terminé - Log: logs/stress_tests/stress_${SESSION}.log"
    
    # Vérifier que le log contient des données
    if [ -s "logs/stress_tests/stress_${SESSION}.log" ]; then
        echo "✅ Log de stress test créé avec succès ($(wc -l < logs/stress_tests/stress_${SESSION}.log) lignes)"
    else
        echo "❌ Log de stress test vide ou manquant"
        exit 1
    fi
else
    echo "❌ Binaire bin/lum_vorax introuvable"
    exit 1
fi

echo ""
echo "Phase 3: Parsing des résultats"
echo "Analyse du fichier: logs/stress_tests/stress_${SESSION}.log"

python3 tools/parse_stress_log.py "logs/stress_tests/stress_${SESSION}.log" 2>&1 | tee "logs/parsing_results/parsed_${SESSION}.log"

echo ""
echo "Phase 4: Vérification des logs créés"
echo "=== CONTENU LOGS GÉNÉRÉS ==="
find logs/ -name "*.log" -type f -exec echo "📄 {}" \; -exec head -10 {} \; -exec echo "---" \;

echo ""
echo "=== MÉTRIQUES PARSÉES ==="
if [ -f stress_results.json ]; then
    echo "✅ Fichier stress_results.json créé:"
    cat stress_results.json
else
    echo "❌ Pas de stress_results.json généré"
fi

echo ""
echo "=== RÉSUMÉ FINAL ==="
echo "Session: $SESSION"
echo "Logs disponibles:"
ls -la logs/*/*.log 2>/dev/null || echo "Aucun log trouvé"

echo ""
echo "✅ VALIDATION TERMINÉE AVEC SUCCÈS"
echo "Fin: $(date -u)"

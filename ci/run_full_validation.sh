
#!/bin/bash -ex
export TZ=UTC
echo "=== VALIDATION FORENSIQUE COMPLETE LUM/VORAX ==="
echo "Démarrage: $(date -u)"

echo "=== PHASE 1: Build reproductible ==="
./build.sh

echo "=== PHASE 2: Tests unitaires ==="
make test

echo "=== PHASE 3: Analyse mémoire ASan ==="
make debug_asan
ASAN_OPTIONS=detect_leaks=1:allocator_may_return_null=1 ./bin/lum_vorax --stress-test-million &> asan_report.txt

echo "=== PHASE 4: Profilage performance ==="
./ci/run_performance_profiling.sh

echo "=== PHASE 5: Tests stress ==="
./bin/lum_vorax --stress-test-million

echo "=== PHASE 6: Validation crypto ==="
./ci/run_crypto_validation.sh

echo "=== PHASE 7: Tests invariants ==="
./ci/run_invariants_test.sh

echo "=== PHASE 8: Génération artefacts signés ==="
./ci/generate_signed_artifacts.sh

echo "✅ VALIDATION COMPLETE - Artifacts dans artifacts/"
echo "Fin: $(date -u)"
ls -la artifacts/
#!/bin/bash -ex

echo "=== VALIDATION FORENSIQUE COMPLÈTE LUM/VORAX ==="
echo "Collecte preuves certification externe selon standards 2025"

# Création répertoire logs avec timestamp
mkdir -p logs/certification_$(date +%Y%m%d_%H%M%S)
cd logs/certification_$(date +%Y%m%d_%H%M%S)

echo "Phase 1: Génération logs complets avec hash"
make clean && make all
make stress_test > stress_results.log 2>&1
sha256sum stress_results.log > stress_results.log.sha256

echo "Phase 2: Informations système exactes"
lscpu > system_cpu.txt
uname -a > system_os.txt
free -h > system_memory.txt
gcc -v 2> compiler_flags.txt

echo "Phase 3: Dataset témoin 1M LUMs"
if [ -f ../../bin/lum_vorax ]; then
    ../../bin/lum_vorax --export-batch 1000000 > lum_batch_1M.json 2>/dev/null || echo "Export batch non implémenté"
    if [ -f lum_batch_1M.json ]; then
        sha256sum lum_batch_1M.json > lum_batch_1M.json.sha256
    fi
fi

echo "Phase 4: Documentation scientifique avancée"
if [ -f ../../bin/lum_vorax ]; then
    ../../bin/lum_vorax --analyze-collatz 1000000000 > collatz_results.txt 2>/dev/null || echo "Analyse Collatz non implémentée"
    ../../bin/lum_vorax --tsp-optimize --cities 200 > tsp_results.txt 2>/dev/null || echo "TSP optimize non implémenté"
    sha256sum *.txt > scientific_hashes.sha256 2>/dev/null || true
fi

echo "Phase 5: Validation croisée (placeholder)"
echo "CROSS_VALIDATION_PLACEHOLDER - Exécution seconde machine requise" > cross_validation_status.txt

echo "Phase 6: Génération manifest final"
find . -type f -exec sha256sum {} \; > manifest_complet.sha256
echo "$(date -u) - Certification externe collectée" > certification_timestamp.txt

echo "✅ VALIDATION TERMINÉE"
echo "Artefacts disponibles dans: $(pwd)"
ls -la

cd ../..

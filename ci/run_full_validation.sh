
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

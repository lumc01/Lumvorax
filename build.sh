
#!/bin/bash -ex
export TZ=UTC
echo "=== BUILD REPRODUCTIBLE LUM/VORAX ==="
echo "Date build: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Environnement: $(uname -a)"
echo "Compilateur: $(clang --version | head -n1)"

# Nettoyage complet
make clean

# Build avec flags reproductibles
make all -j$(nproc)

echo "✅ Build terminé - binaire dans bin/lum_vorax"
ls -la bin/lum_vorax

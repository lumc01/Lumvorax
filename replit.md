# Simulation de Trou Noir - Nouvelle Génération

## Architecture
- **Physique**: Coordonnées de Kerr-Schild pour franchissement d'horizon.
- **Numérique**: Quad Precision (float128) via libquadmath.
- **Validation**: Suivi strict des contraintes d'Einstein et de Carter.

## Objectifs T21-T27
1. Stabilité < 1e-15 sur 1M de pas.
2. Analyse de scaling (1-a)^n.
3. Spectre de Lyapunov rigoureux.
4. Couplage MHD sans artefact de coordonnée.

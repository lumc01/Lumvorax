# Comparaison Avant/Après — Cycle 06

Run ID: `research_20260309T194824Z_6894`

## Avant
- Contrôles plasma partiels et métadonnées physiques incomplètes.
- Pas de table dédiée aux paramètres (U/t, dopage, BC, Δt, jauge).
- Pas de suite explicite Von Neumann/cas jouet.

## Après
- Contrôles phase+pump+quench actifs, stabilité longue et sweep Δt conservés.
- Pompage dynamique actif et tracé contre une trajectoire sans contrôle.
- `module_physics_metadata.csv` ajouté (lattice, U/t, dopage, BC, schéma, Δt, jauge, β, volume, type de champ) pour 13 modules.
- `normalized_observables_trace.csv` ajouté (énergie/pairing normalisés).
- `numerical_stability_suite.csv` + `toy_model_validation.csv` ajoutés.

## Différences quantitatives clés
- FFT dominant_freq=0.0244140625, dominant_amp=110.1198604011.
- Feedback energy_reduction_ratio=-12.1988123712, pairing_gain=-0.2516922036.
- Drift max énergie QF (2200/4400/6600/8800)=0.0000078669 (PASS).
- Rayon spectral Von Neumann (hubbard_hts_core)=1.0002246148 (FAIL).
- Cas jouet exp_decay abs_error=0.0000021287 (PASS).

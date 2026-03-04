# Rapport pédagogique - Hubbard HTS et problèmes comparables

Run ID: `1772585474_4175`

- Log principal: `/workspace/Lumvorax/src/advanced_calculations/quantum_problem_hubbard_hts/results/1772585474_4175/logs/execution.log`
- Métriques CSV: `/workspace/Lumvorax/src/advanced_calculations/quantum_problem_hubbard_hts/results/1772585474_4175/logs/metrics.csv`
- Snapshot hardware: `/workspace/Lumvorax/src/advanced_calculations/quantum_problem_hubbard_hts/results/1772585474_4175/logs/hardware_snapshot.log`

## Résumé simple
Le module exécute 5 problèmes l'un après l'autre, sans écraser les anciens résultats.
- **hubbard_hts_core**: énergie=49049.8030, pairing=0.869113, ratio de signe=-0.003304, CPU max=14.89%, RAM max=3.35%, temps=989223313ns.
- **qcd_lattice_proxy**: énergie=19709.2859, pairing=0.847580, ratio de signe=0.000283, CPU max=14.94%, RAM max=3.35%, temps=722965665ns.
- **quantum_field_noneq**: énergie=-721.4313, pairing=0.801240, ratio de signe=0.003624, CPU max=14.99%, RAM max=3.35%, temps=605555000ns.
- **dense_nuclear_proxy**: énergie=26026.3068, pairing=0.908758, ratio de signe=-0.005696, CPU max=15.03%, RAM max=3.35%, temps=620872196ns.
- **quantum_chemistry_proxy**: énergie=-7064.9259, pairing=0.925048, ratio de signe=-0.008067, CPU max=15.08%, RAM max=3.33%, temps=675565096ns.

## Interprétation vulgarisée
- Plus le `pairing` est élevé, plus les électrons tendent à former des paires coopératives.
- Un `ratio de signe` proche de 0 montre la difficulté numérique type *sign problem*.
- Les lignes numérotées du log `execution.log` permettent de référencer précisément les évènements.

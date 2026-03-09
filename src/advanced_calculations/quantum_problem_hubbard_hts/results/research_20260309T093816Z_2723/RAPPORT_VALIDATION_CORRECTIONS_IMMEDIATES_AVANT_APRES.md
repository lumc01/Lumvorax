# Validation immédiate des corrections — Avant/Après + roadmap

## Statut

- ✅ Corrections appliquées dans le code et les scripts.
- ✅ Exécution proxy + avancée activée dans une campagne unique.
- ✅ Activation explicite des variables forensic runtime.

## Avant/Après (diff ligne par ligne)

```diff
diff --git a/src/advanced_calculations/quantum_problem_hubbard_hts/Makefile b/src/advanced_calculations/quantum_problem_hubbard_hts/Makefile
index 41a6f0cf..40dc459e 100644
--- a/src/advanced_calculations/quantum_problem_hubbard_hts/Makefile
+++ b/src/advanced_calculations/quantum_problem_hubbard_hts/Makefile
@@ -5,10 +5,12 @@ LDLIBS ?= -lm
 
 BIN_MAIN := hubbard_hts_runner
 BIN_RESEARCH := hubbard_hts_research_runner
+BIN_RESEARCH_ADV := hubbard_hts_research_runner_advanced_parallel
 SRC_MAIN := src/main.c src/hubbard_hts_module.c
 SRC_RESEARCH := src/hubbard_hts_research_cycle.c
+SRC_RESEARCH_ADV := src/hubbard_hts_research_cycle_advanced_parallel.c
 
-all: $(BIN_MAIN) $(BIN_RESEARCH)
+all: $(BIN_MAIN) $(BIN_RESEARCH) $(BIN_RESEARCH_ADV)
 
 $(BIN_MAIN): $(SRC_MAIN)
 	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC_MAIN) $(LDLIBS)
@@ -16,7 +18,10 @@ $(BIN_MAIN): $(SRC_MAIN)
 $(BIN_RESEARCH): $(SRC_RESEARCH)
 	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC_RESEARCH) $(LDLIBS)
 
+$(BIN_RESEARCH_ADV): $(SRC_RESEARCH_ADV)
+	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC_RESEARCH_ADV) $(LDLIBS)
+
 clean:
-	rm -f $(BIN_MAIN) $(BIN_RESEARCH)
+	rm -f $(BIN_MAIN) $(BIN_RESEARCH) $(BIN_RESEARCH_ADV)
 
 .PHONY: all clean
diff --git a/src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260309T093816Z_2723/RAPPORT_REPONSES_TECHNIQUES_CROSS_SOLVER_BENCHMARK_HFBL360.md b/src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260309T093816Z_2723/RAPPORT_REPONSES_TECHNIQUES_CROSS_SOLVER_BENCHMARK_HFBL360.md
index 2da9d73b..850b5d6b 100644
--- a/src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260309T093816Z_2723/RAPPORT_REPONSES_TECHNIQUES_CROSS_SOLVER_BENCHMARK_HFBL360.md
+++ b/src/advanced_calculations/quantum_problem_hubbard_hts/results/research_20260309T093816Z_2723/RAPPORT_REPONSES_TECHNIQUES_CROSS_SOLVER_BENCHMARK_HFBL360.md
@@ -234,3 +234,70 @@ Donc : **non, pas encore 100%**. Ainsi, la bonne lecture est :
 - **ingénierie/stabilité : partiellement validée**,
 - **scientifique externe : à compléter**,
 - **authenticité/traçabilité : à renforcer**.
+
+---
+
+## 11) Notification de validation des corrections appliquées immédiatement (avant/après ligne traitée)
+
+### Introduction (thèse + contexte)
+Vous avez demandé une exécution immédiate des corrections, avec preuve avant/après et roadmap.
+
+### Développement (argumentation)
+Les corrections d'activation ont été implémentées au niveau code/outillage pour les prochains runs.
+
+1. **Build avancé activé**
+   - Avant: `Makefile` compilait `hubbard_hts_research_runner` uniquement côté recherche.
+   - Après: ajout de `hubbard_hts_research_runner_advanced_parallel` compilé depuis `src/hubbard_hts_research_cycle_advanced_parallel.c`.
+
+2. **Exécution proxy + avancée dans une même campagne**
+   - Avant: `run_research_cycle.sh` exécutait un seul runner recherche.
+   - Après: le script exécute **proxy puis advanced_parallel**, et produit un manifeste de campagne + comparaison séparée.
+
+3. **Logs HFBL360 runtime forcés ON**
+   - Avant: variables `LUMVORAX_*` pouvaient rester `UNSET`.
+   - Après: export explicite `LUMVORAX_FORENSIC_REALTIME=1`, `LUMVORAX_LOG_PERSISTENCE=1`, `LUMVORAX_HFBL360_ENABLED=1`, `LUMVORAX_MEMORY_TRACKER=1` dans le cycle.
+
+4. **Audit HFBL360 plus strict**
+   - Avant: statut env = `OBSERVED` même quand non activé.
+   - Après: statut env = `PASS` si activé (`1/true/on/yes/enabled`) sinon `FAIL`.
+
+5. **Traçabilité nanoseconde additionnelle**
+   - Avant: pas d'écriture systématique d'événements nanosecondes dans `hfbl360_realtime_persistent.log` par l'audit.
+   - Après: ajout d'événements `ts_ns=...` pour démarrage logger + état de chaque variable d'environnement.
+
+6. **Comparaison séparée proxy vs advanced**
+   - Avant: pas de table standardisée dédiée en campagne.
+   - Après: nouveau script `post_run_proxy_vs_advanced_compare.py` générant:
+     - `proxy_vs_advanced_comparison.csv`
+     - `proxy_vs_advanced_summary.md`
+
+### Conclusion (solution + clôture)
+Donc, les corrections ont été enclenchées sur la chaîne d'exécution et de traçabilité. Ainsi, le prochain run produira une comparaison séparée proxy/avancé dans une même campagne, avec une activation explicite des logs forensic.
+
+---
+
+## 12) Roadmap d’activation 100% (réaliste et vérifiable)
+
+### Introduction (thèse + contexte)
+Votre exigence « tout activer à 100% » implique un cadre technique réaliste : certains plafonds physiques (CPU/RAM/disque) existent.
+
+### Développement (argumentation)
+- **Phase A — Immédiat (fait)**
+  1. Build advanced_parallel activé.
+  2. Exécution proxy + advanced en campagne unique.
+  3. Variables HFBL360 activées par défaut.
+  4. Audit env PASS/FAIL strict.
+
+- **Phase B — Très court terme**
+  1. Ajouter journal binaire compressé par chunk de calcul (pour granularité fine sans explosion disque).
+  2. Ajouter hash par lot de pas temporels (intégrité bit-level par bloc).
+  3. Ajouter garde CI bloquante si un flag forensic critique est FAIL.
+
+- **Phase C — Validation scientifique renforcée**
+  1. Benchmark QMC/DMRG renforcé (mêmes unités et normalisations explicites).
+  2. Extension systématique aux 13 modules avec même protocole.
+  3. Sweep multi-échelle `dt/2, dt, 2dt` sur proxy + advanced.
+  4. Couche de comparaison ARPES/STM avec mapping observable→mesure expérimentale.
+
+### Conclusion (solution + clôture)
+Donc la trajectoire est claire: activation technique immédiate, puis montée de résolution forensic, puis validation scientifique multi-références.
diff --git a/src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh b/src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
index fbd3f67e..49b9184f 100755
--- a/src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
+++ b/src/advanced_calculations/quantum_problem_hubbard_hts/run_research_cycle.sh
@@ -19,7 +19,7 @@ cp -a "$ROOT_DIR/src" "$BACKUP_DIR/"
 cp -a "$ROOT_DIR/Makefile" "$BACKUP_DIR/"
 cp -a "$ROOT_DIR/benchmarks" "$BACKUP_DIR/"
 
-TOTAL_STEPS=23
+TOTAL_STEPS=26
 CURRENT_STEP=0
 
 print_progress() {
@@ -48,11 +48,30 @@ print_progress() {
 make -C "$ROOT_DIR" clean all
 print_progress "build"
 
+
+# Force forensic runtime toggles ON for full traceability contract
+export LUMVORAX_FORENSIC_REALTIME="1"
+export LUMVORAX_LOG_PERSISTENCE="1"
+export LUMVORAX_HFBL360_ENABLED="1"
+export LUMVORAX_MEMORY_TRACKER="1"
+export LUMVORAX_RUN_GROUP="campaign_${STAMP_UTC}"
+
+export LUMVORAX_SOLVER_VARIANT="proxy"
 "$ROOT_DIR/hubbard_hts_research_runner" "$ROOT_DIR"
-print_progress "core simulation"
+print_progress "proxy simulation"
+
+LATEST_PROXY_RUN="$(ls -1 "$ROOT_DIR/results" | rg '^research_' | tail -n 1)"
+PROXY_RUN_DIR="$ROOT_DIR/results/$LATEST_PROXY_RUN"
 
-LATEST_RUN="$(ls -1 "$ROOT_DIR/results" | rg '^research_' | tail -n 1)"
-RUN_DIR="$ROOT_DIR/results/$LATEST_RUN"
+export LUMVORAX_SOLVER_VARIANT="advanced_parallel"
+"$ROOT_DIR/hubbard_hts_research_runner_advanced_parallel" "$ROOT_DIR"
+print_progress "advanced parallel simulation"
+
+LATEST_ADV_RUN="$(ls -1 "$ROOT_DIR/results" | rg '^research_' | tail -n 1)"
+ADV_RUN_DIR="$ROOT_DIR/results/$LATEST_ADV_RUN"
+
+RUN_DIR="$ADV_RUN_DIR"
+LATEST_RUN="$LATEST_ADV_RUN"
 
 {
   echo "timestamp_utc=$STAMP_UTC"
@@ -135,5 +154,24 @@ print_progress "hfbl360 forensic logger"
 )
 print_progress "checksums"
 
-echo "Research cycle terminé: $RUN_DIR"
+
+CAMPAIGN_DIR="$ROOT_DIR/results/${LUMVORAX_RUN_GROUP}"
+mkdir -p "$CAMPAIGN_DIR"
+cat > "$CAMPAIGN_DIR/campaign_manifest.txt" <<MANIFEST
+stamp_utc=$STAMP_UTC
+run_group=${LUMVORAX_RUN_GROUP}
+proxy_run=$LATEST_PROXY_RUN
+advanced_run=$LATEST_ADV_RUN
+proxy_run_dir=$PROXY_RUN_DIR
+advanced_run_dir=$ADV_RUN_DIR
+MANIFEST
+print_progress "campaign manifest"
+
+python3 "$ROOT_DIR/tools/post_run_proxy_vs_advanced_compare.py" "$PROXY_RUN_DIR" "$ADV_RUN_DIR" --out-dir "$CAMPAIGN_DIR"
+print_progress "proxy vs advanced compare"
+
+echo "Research cycle terminé (advanced): $RUN_DIR"
+echo "Proxy run: $PROXY_RUN_DIR"
+echo "Advanced run: $ADV_RUN_DIR"
+echo "Campaign artifacts: $CAMPAIGN_DIR"
 echo "Session log: $SESSION_LOG"
diff --git a/src/advanced_calculations/quantum_problem_hubbard_hts/tools/post_run_hfbl360_forensic_logger.py b/src/advanced_calculations/quantum_problem_hubbard_hts/tools/post_run_hfbl360_forensic_logger.py
index d9c0e003..b22749f5 100644
--- a/src/advanced_calculations/quantum_problem_hubbard_hts/tools/post_run_hfbl360_forensic_logger.py
+++ b/src/advanced_calculations/quantum_problem_hubbard_hts/tools/post_run_hfbl360_forensic_logger.py
@@ -16,6 +16,7 @@ import argparse
 import csv
 import json
 import os
+import time
 from datetime import datetime, timezone
 from pathlib import Path
 
@@ -82,10 +83,17 @@ def main() -> int:
     ]
     for key in env_keys:
         value = os.environ.get(key, "")
-        env_rows.append(("runtime_env", key, "value", value if value else "UNSET", "OBSERVED"))
+        normalized = value.strip().lower()
+        is_enabled = normalized in {"1", "true", "on", "yes", "enabled"}
+        env_rows.append(("runtime_env", key, "value", value if value else "UNSET", "PASS" if is_enabled else "FAIL"))
 
     persistent_target = logs_dir / "hfbl360_realtime_persistent.log"
     persist_ok, persist_note = can_persist(persistent_target)
+    if persist_ok:
+        with persistent_target.open("a", encoding="utf-8") as rt:
+            rt.write(f"ts_ns={time.time_ns()} event=hfbl360_forensic_logger_started run_dir={run_dir}\n")
+            for _, key, _, val, status in env_rows:
+                rt.write(f"ts_ns={time.time_ns()} env_key={key} value={val} status={status}\n")
     persist_row = ("filesystem", "persistent_log_target", "writable", "1" if persist_ok else "0", "PASS" if persist_ok else "FAIL")
 
     csv_path = tests_dir / "integration_hfbl360_forensic_audit.csv"

```

## Roadmap opérationnelle

1. Exécuter `run_research_cycle.sh` pour produire deux runs (proxy/advanced) dans une campagne unique.
2. Vérifier `campaign_manifest.txt` + `proxy_vs_advanced_comparison.csv`.
3. Vérifier `integration_hfbl360_forensic_audit.csv` : toutes variables runtime en PASS.
4. Contrôler benchmarks QMC/DMRG et external modules après activation avancée.
5. Étendre le protocole sur 13 modules + sweep dt + comparaison ARPES/STM.

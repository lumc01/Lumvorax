#ifndef ULTRA_FORENSIC_LOGGER_H_
#define ULTRA_FORENSIC_LOGGER_H_

/*
 * ultra_forensic_logger.h — LumVorax C59-ULTRA
 * Traçabilité 100% : ZÉRO filtre, ZÉRO throttle, ZÉRO réduction.
 * Chaque appel FORENSIC_LOG_MODULE_METRIC et FORENSIC_LOG_NANO
 * écrit immédiatement dans le CSV avec fflush — aucune entrée n'est perdue.
 * Thread-safe via pthread_mutex.
 */

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Timestamp nanoseconde haute résolution (CLOCK_MONOTONIC) ── */
static inline uint64_t now_ns(void) {
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    return (uint64_t)_ts.tv_sec * UINT64_C(1000000000)
         + (uint64_t)_ts.tv_nsec;
}

/* ── API publique ── */

/*
 * Ouvre le fichier CSV de log LumVorax forensique.
 * Écrit l'en-tête CSV.  Doit être appelé AVANT tout FORENSIC_LOG_*.
 */
void ultra_forensic_logger_init_lum(const char* csv_path);

/*
 * Ferme le fichier CSV — flush final garanti avant fclose.
 */
void ultra_forensic_logger_destroy(void);

/*
 * Écrit une entrée CSV : seq,timestamp_ns,module,key,value
 * avec fflush immédiat — ZÉRO perte.
 */
void ultra_forensic_logger_write(const char* module,
                                  const char* key,
                                  double      value);

/* ── Macros sans filtre ── */

/*
 * FORENSIC_LOG_MODULE_METRIC — log de métrique par module.
 * Chaque appel → une ligne CSV immédiate.  ZÉRO filtre.
 */
#define FORENSIC_LOG_MODULE_METRIC(mod, key, val) \
    ultra_forensic_logger_write((mod), (key), (double)(val))

/*
 * FORENSIC_LOG_NANO — log sweep-par-sweep nanoseconde.
 * Même comportement que FORENSIC_LOG_MODULE_METRIC : chaque appel →
 * une ligne CSV immédiate avec timestamp nanoseconde.  ZÉRO filtre.
 */
#define FORENSIC_LOG_NANO(mod, key, val) \
    ultra_forensic_logger_write((mod), (key), (double)(val))

#ifdef __cplusplus
}
#endif

#endif /* ULTRA_FORENSIC_LOGGER_H_ */

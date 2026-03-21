#define _GNU_SOURCE
/*
 * ultra_forensic_logger.c — LumVorax C59-ULTRA
 * Implémentation thread-safe du logger forensique.
 * RÈGLE ABSOLUE : ZÉRO filtre, ZÉRO throttle, ZÉRO réduction.
 * Chaque appel à ultra_forensic_logger_write() produit
 * une ligne CSV et un fflush() immédiat.
 * SHA-256/SHA-512 calculés par run_research_cycle.sh sur le CSV final.
 */

#include "ultra_forensic_logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>

static FILE*           g_ufl_fp   = NULL;
static pthread_mutex_t g_ufl_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t        g_ufl_seq  = 0;

void ultra_forensic_logger_init_lum(const char* csv_path) {
    pthread_mutex_lock(&g_ufl_lock);
    if (g_ufl_fp) {
        fflush(g_ufl_fp);
        fclose(g_ufl_fp);
        g_ufl_fp = NULL;
    }
    g_ufl_fp = fopen(csv_path, "w");
    if (!g_ufl_fp) {
        fprintf(stderr,
            "[UFL-ERROR] Impossible d'ouvrir le log LumVorax: %s (errno=%d: %s)\n",
            csv_path, errno, strerror(errno));
        pthread_mutex_unlock(&g_ufl_lock);
        return;
    }
    /* En-tête CSV — colonnes fixes pour parsing automatique */
    fprintf(g_ufl_fp, "seq,timestamp_ns,module,key,value\n");
    fflush(g_ufl_fp);
    g_ufl_seq = 0;
    pthread_mutex_unlock(&g_ufl_lock);
    fprintf(stderr,
        "[UFL-OK] LumVorax forensic log ouvert: %s (ZERO FILTRE — 100%% capture)\n",
        csv_path);
}

void ultra_forensic_logger_destroy(void) {
    pthread_mutex_lock(&g_ufl_lock);
    if (g_ufl_fp) {
        fflush(g_ufl_fp);
        fclose(g_ufl_fp);
        g_ufl_fp = NULL;
        fprintf(stderr, "[UFL-OK] LumVorax forensic log fermé (flush final garanti)\n");
    }
    pthread_mutex_unlock(&g_ufl_lock);
}

void ultra_forensic_logger_write(const char* module,
                                  const char* key,
                                  double      value) {
    /* Timestamp nanoseconde CLOCK_MONOTONIC — haute résolution */
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t ns = (uint64_t)_ts.tv_sec * UINT64_C(1000000000)
                + (uint64_t)_ts.tv_nsec;

    pthread_mutex_lock(&g_ufl_lock);
    if (g_ufl_fp) {
        /* Format CSV : seq,timestamp_ns,module,key,value */
        fprintf(g_ufl_fp,
            "%" PRIu64 ",%" PRIu64 ",%s,%s,%.17g\n",
            ++g_ufl_seq,
            ns,
            module ? module : "",
            key    ? key    : "",
            value);
        /* fflush immédiat — ZÉRO perte même en cas d'interruption */
        fflush(g_ufl_fp);
    }
    pthread_mutex_unlock(&g_ufl_lock);
}

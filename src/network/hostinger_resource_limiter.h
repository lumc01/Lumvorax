
#ifndef HOSTINGER_RESOURCE_LIMITER_H
#define HOSTINGER_RESOURCE_LIMITER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Fonctions vérification ressources serveur Hostinger
bool hostinger_check_cpu_availability(void);
bool hostinger_check_ram_availability(size_t required_mb);
bool hostinger_check_lum_processing_limit(size_t lum_count);

// Gestion monitor ressources
typedef struct hostinger_resource_monitor_t hostinger_resource_monitor_t;

hostinger_resource_monitor_t* hostinger_resource_monitor_create(void);
void hostinger_resource_monitor_destroy(void);

// Constantes limites serveur
#define HOSTINGER_CPU_CORES 2
#define HOSTINGER_RAM_GB 6  // Sur 7.8GB total, garder marge sécurité
#define HOSTINGER_STORAGE_GB 90  // Sur 100GB total
#define HOSTINGER_MAX_LUMS_CONCURRENT 1000000

#endif // HOSTINGER_RESOURCE_LIMITER_H

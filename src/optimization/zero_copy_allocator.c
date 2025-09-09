#include "zero_copy_allocator.h"
#include "../logger/lum_logger.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <stdio.h>

// Configuration et constantes
#define DEFAULT_ALIGNMENT 64
#define MMAP_REGION_PREFIX "/tmp/lum_zero_copy_"
#define MAX_FREE_BLOCKS 1024

static uint64_t next_allocation_id = 1;

static uint64_t get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL;
}

zero_copy_pool_t* zero_copy_pool_create(size_t size, const char* name) {
    zero_copy_pool_t* pool = malloc(sizeof(zero_copy_pool_t));
    if (!pool) {
        lum_log(LUM_LOG_ERROR, "Failed to allocate zero_copy_pool_t structure");
        return NULL;
    }

    // Configuration initiale
    pool->total_size = size;
    pool->used_size = 0;
    pool->alignment = DEFAULT_ALIGNMENT;
    pool->is_mmap_backed = false;
    pool->mmap_fd = -1;
    pool->free_list = NULL;
    pool->free_blocks_count = 0;
    pool->allocations_served = 0;
    pool->zero_copy_hits = 0;
    pool->memory_reused_bytes = 0;

    // Nom de région pour debugging
    pool->region_name = malloc(strlen(name) + 1);
    if (pool->region_name) {
        strcpy(pool->region_name, name);
    }

    // Allocation initiale (standard malloc, upgradée à mmap plus tard si demandé)
    pool->memory_region = malloc(size);
    if (!pool->memory_region) {
        lum_log(LUM_LOG_ERROR, "Failed to allocate %zu bytes for zero-copy pool", size);
        free(pool->region_name);
        free(pool);
        return NULL;
    }

    // Initialisation à zéro pour sécurité
    memset(pool->memory_region, 0, size);

    lum_log(LUM_LOG_INFO, "Zero-copy pool '%s' created: %zu bytes", name, size);
    return pool;
}

void zero_copy_pool_destroy(zero_copy_pool_t* pool) {
    if (!pool) return;

    lum_log(LUM_LOG_INFO, "Destroying zero-copy pool '%s'", 
            pool->region_name ? pool->region_name : "unnamed");

    // Statistics finales
    zero_copy_print_stats(pool);

    // Nettoyage de la free list
    free_block_t* current = pool->free_list;
    while (current) {
        free_block_t* next = current->next;
        free(current);
        current = next;
    }

    // Libération mémoire selon le type
    if (pool->is_mmap_backed && pool->memory_region != MAP_FAILED) {
        if (munmap(pool->memory_region, pool->total_size) != 0) {
            lum_log(LUM_LOG_ERROR, "munmap failed: %s", strerror(errno));
        }
        if (pool->mmap_fd >= 0) {
            close(pool->mmap_fd);
        }
    } else {
        free(pool->memory_region);
    }

    free(pool->region_name);
    free(pool);
}

zero_copy_allocation_t* zero_copy_alloc(zero_copy_pool_t* pool, size_t size) {
    if (!pool || size == 0) return NULL;

    // Aligner la taille sur la granularité du pool
    size_t aligned_size = (size + pool->alignment - 1) & ~(pool->alignment - 1);

    zero_copy_allocation_t* allocation = malloc(sizeof(zero_copy_allocation_t));
    if (!allocation) return NULL;

    allocation->size = aligned_size;
    allocation->is_zero_copy = false;
    allocation->is_reused_memory = false;
    allocation->allocation_id = next_allocation_id++;
    allocation->ptr = NULL;

    // Phase 1: Tentative de réutilisation zero-copy depuis free list
    free_block_t* prev = NULL;
    free_block_t* current = pool->free_list;

    while (current) {
        if (current->size >= aligned_size) {
            // Found reusable block - ZERO COPY!
            allocation->ptr = current->ptr;
            allocation->is_zero_copy = true;
            allocation->is_reused_memory = true;

            // Retirer de la free list
            if (prev) {
                prev->next = current->next;
            } else {
                pool->free_list = current->next;
            }

            pool->free_blocks_count--;
            pool->zero_copy_hits++;
            pool->memory_reused_bytes += aligned_size;

            // Mise à jour statistiques
            uint64_t reuse_time = get_timestamp_us() - current->last_used_timestamp;
            lum_log(LUM_LOG_DEBUG, "Zero-copy reuse: %zu bytes, reused after %lu μs", 
                    aligned_size, reuse_time);

            free(current);
            pool->allocations_served++;
            return allocation;
        }
        prev = current;
        current = current->next;
    }

    // Phase 2: Allocation dans la région principale
    if (pool->used_size + aligned_size <= pool->total_size) {
        allocation->ptr = (uint8_t*)pool->memory_region + pool->used_size;
        pool->used_size += aligned_size;
        allocation->is_zero_copy = true; // Techniquement zero-copy car pas de memcpy

        lum_log(LUM_LOG_DEBUG, "Pool allocation: %zu bytes at offset %zu", 
                aligned_size, pool->used_size - aligned_size);

        pool->allocations_served++;
        return allocation;
    }

    // Phase 3: Fallback allocation standard (non zero-copy)
    allocation->ptr = malloc(aligned_size);
    if (!allocation->ptr) {
        free(allocation);
        return NULL;
    }

    lum_log(LUM_LOG_DEBUG, "Standard allocation (pool full): %zu bytes", aligned_size);
    pool->allocations_served++;
    return allocation;
}

bool zero_copy_free(zero_copy_pool_t* pool, zero_copy_allocation_t* allocation) {
    if (!pool || !allocation || !allocation->ptr) return false;

    // Si c'est une allocation zero-copy dans le pool, ajouter à la free list
    if (allocation->is_zero_copy && 
        (uint8_t*)allocation->ptr >= (uint8_t*)pool->memory_region && 
        (uint8_t*)allocation->ptr < (uint8_t*)pool->memory_region + pool->total_size) {

        // Créer un nouveau free block
        free_block_t* free_block = malloc(sizeof(free_block_t));
        if (free_block) {
            free_block->ptr = allocation->ptr;
            free_block->size = allocation->size;
            free_block->last_used_timestamp = get_timestamp_us();
            free_block->next = pool->free_list;

            pool->free_list = free_block;
            pool->free_blocks_count++;

            // Compaction si nécessaire
            if (pool->free_blocks_count > MAX_FREE_BLOCKS) {
                zero_copy_compact_free_list(pool);
            }

            lum_log(LUM_LOG_DEBUG, "Added to free list: %zu bytes (total free blocks: %zu)", 
                    allocation->size, pool->free_blocks_count);
        }
    } else {
        // Allocation standard - libérer directement
        free(allocation->ptr);
    }

    return true;
}

bool zero_copy_resize_inplace(zero_copy_pool_t* pool, zero_copy_allocation_t* allocation, size_t new_size) {
    if (!pool || !allocation || !allocation->ptr) return false;

    size_t aligned_new_size = (new_size + pool->alignment - 1) & ~(pool->alignment - 1);

    // Resize impossible pour allocations non-pool
    if (!allocation->is_zero_copy) return false;

    // Si dans le pool et au bout de la région utilisée, possible resize
    if (allocation->ptr >= pool->memory_region &&
        (uint8_t*)allocation->ptr + allocation->size == (uint8_t*)pool->memory_region + pool->used_size) {

        // Expansion possible ?
        if (aligned_new_size > allocation->size) {
            size_t additional = aligned_new_size - allocation->size;
            if (pool->used_size + additional <= pool->total_size) {
                pool->used_size += additional;
                allocation->size = aligned_new_size;

                lum_log(LUM_LOG_DEBUG, "In-place expansion: %zu -> %zu bytes", 
                        allocation->size, aligned_new_size);
                return true;
            }
        } else {
            // Contraction toujours possible
            size_t reduction = allocation->size - aligned_new_size;
            pool->used_size -= reduction;
            allocation->size = aligned_new_size;

            lum_log(LUM_LOG_DEBUG, "In-place contraction: %zu -> %zu bytes", 
                    allocation->size, aligned_new_size);
            return true;
        }
    }

    return false;
}

bool zero_copy_enable_mmap_backing(zero_copy_pool_t* pool) {
    if (!pool || pool->is_mmap_backed) return false;

    // Créer fichier temporaire pour mapping
    char temp_path[256];
    snprintf(temp_path, sizeof(temp_path), "%s%s_%d", 
             MMAP_REGION_PREFIX, 
             pool->region_name ? pool->region_name : "pool", 
             getpid());

    int fd = open(temp_path, O_CREAT | O_RDWR | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        lum_log(LUM_LOG_ERROR, "Failed to create mmap file: %s", strerror(errno));
        return false;
    }

    // Étendre le fichier à la taille du pool
    if (ftruncate(fd, pool->total_size) != 0) {
        lum_log(LUM_LOG_ERROR, "Failed to resize mmap file: %s", strerror(errno));
        close(fd);
        unlink(temp_path);
        return false;
    }

    // Memory mapping
    void* mapped = mmap(NULL, pool->total_size, PROT_READ | PROT_WRITE, 
                       MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        lum_log(LUM_LOG_ERROR, "mmap failed: %s", strerror(errno));
        close(fd);
        unlink(temp_path);
        return false;
    }

    // Copier les données existantes si nécessaire
    if (pool->used_size > 0) {
        memcpy(mapped, pool->memory_region, pool->used_size);
    }

    // Basculer vers mmap
    free(pool->memory_region);
    pool->memory_region = mapped;
    pool->is_mmap_backed = true;
    pool->mmap_fd = fd;

    // Nettoyer fichier temporaire (reste mappé en mémoire)
    unlink(temp_path);

    lum_log(LUM_LOG_INFO, "Zero-copy pool '%s' upgraded to mmap backing", 
            pool->region_name);
    return true;
}

bool zero_copy_prefault_pages(zero_copy_pool_t* pool) {
    if (!pool || !pool->is_mmap_backed) return false;

    // Prefault toutes les pages en les touchant
    size_t page_size = getpagesize();
    uint8_t* mem = (uint8_t*)pool->memory_region;

    for (size_t offset = 0; offset < pool->total_size; offset += page_size) {
        // Touch each page to trigger page fault now
        volatile uint8_t touch = mem[offset];
        (void)touch; // Éviter warning unused
    }

    lum_log(LUM_LOG_DEBUG, "Prefaulted %zu pages for zero-copy pool", 
            pool->total_size / page_size);
    return true;
}

bool zero_copy_advise_sequential(zero_copy_pool_t* pool) {
    if (!pool || !pool->is_mmap_backed) return false;

    // Optimisation pour accès séquentiel
    if (madvise(pool->memory_region, pool->total_size, MADV_SEQUENTIAL) == 0) {
        lum_log(LUM_LOG_DEBUG, "Sequential access advised for zero-copy pool");
        return true;
    }

    lum_log(LUM_LOG_ERROR, "madvise SEQUENTIAL failed: %s", strerror(errno));
    return false;
}

void zero_copy_print_stats(zero_copy_pool_t* pool) {
    if (!pool) return;

    double efficiency = zero_copy_get_efficiency_ratio(pool);
    size_t fragmentation = zero_copy_get_fragmentation_bytes(pool);

    printf("=== Zero-Copy Pool Statistics '%s' ===\n", 
           pool->region_name ? pool->region_name : "unnamed");
    printf("Pool size: %zu bytes (%.2f MB)\n", 
           pool->total_size, pool->total_size / (1024.0 * 1024.0));
    printf("Used size: %zu bytes (%.2f%% utilization)\n", 
           pool->used_size, (double)pool->used_size / pool->total_size * 100.0);
    printf("Backing: %s\n", pool->is_mmap_backed ? "mmap" : "malloc");
    printf("Alignment: %zu bytes\n", pool->alignment);
    printf("Allocations served: %zu\n", pool->allocations_served);
    printf("Zero-copy hits: %zu (%.2f%%)\n", 
           pool->zero_copy_hits, 
           pool->allocations_served > 0 ? 
           (double)pool->zero_copy_hits / pool->allocations_served * 100.0 : 0.0);
    printf("Memory reused: %zu bytes (%.2f MB)\n", 
           pool->memory_reused_bytes, pool->memory_reused_bytes / (1024.0 * 1024.0));
    printf("Free blocks: %zu\n", pool->free_blocks_count);
    printf("Efficiency ratio: %.3f\n", efficiency);
    printf("Fragmentation: %zu bytes\n", fragmentation);
    printf("=========================================\n");
}

double zero_copy_get_efficiency_ratio(zero_copy_pool_t* pool) {
    if (!pool || pool->allocations_served == 0) return 0.0;
    return (double)pool->zero_copy_hits / pool->allocations_served;
}

size_t zero_copy_get_fragmentation_bytes(zero_copy_pool_t* pool) {
    if (!pool) return 0;

    size_t total_free = 0;
    free_block_t* current = pool->free_list;

    while (current) {
        total_free += current->size;
        current = current->next;
    }

    return total_free;
}

bool zero_copy_defragment_pool(zero_copy_pool_t* pool) {
    if (!pool || pool->free_blocks_count == 0) return false;

    lum_log(LUM_LOG_INFO, "Starting defragmentation of zero-copy pool");

    // Pour un défragmentation complète, il faudrait:
    // 1. Identifier tous les blocs alloués
    // 2. Les compacter vers le début
    // 3. Mettre à jour tous les pointeurs
    // 
    // C'est complexe et dangereux sans GC. Pour l'instant, on fait une
    // compaction simple de la free list.

    return zero_copy_compact_free_list(pool);
}

bool zero_copy_compact_free_list(zero_copy_pool_t* pool) {
    if (!pool || pool->free_blocks_count <= 1) return false;

    size_t original_count = pool->free_blocks_count;

    // Trier les blocs libres par adresse pour détecter les adjacents
    free_block_t* sorted_list = pool->free_list;
    pool->free_list = NULL;
    pool->free_blocks_count = 0;

    // Algorithme simple: réinsérer en triant et fusionner les adjacents
    while (sorted_list) {
        free_block_t* current = sorted_list;
        sorted_list = sorted_list->next;

        // Trouver position d'insertion
        free_block_t* prev = NULL;
        free_block_t* pos = pool->free_list;

        while (pos && pos->ptr < current->ptr) {
            prev = pos;
            pos = pos->next;
        }

        // Insérer
        current->next = pos;
        if (prev) {
            prev->next = current;
        } else {
            pool->free_list = current;
        }
        pool->free_blocks_count++;

        // Fusion avec le précédent si adjacent
        if (prev && (uint8_t*)prev->ptr + prev->size == current->ptr) {
            prev->size += current->size;
            prev->next = current->next;
            free(current);
            pool->free_blocks_count--;
            current = prev; // Pour fusion avec suivant
        }

        // Fusion avec le suivant si adjacent
        if (current->next && (uint8_t*)current->ptr + current->size == current->next->ptr) {
            free_block_t* next = current->next;
            current->size += next->size;
            current->next = next->next;
            free(next);
            pool->free_blocks_count--;
        }
    }

    size_t compacted = original_count - pool->free_blocks_count;
    if (compacted > 0) {
        lum_log(LUM_LOG_INFO, "Free list compaction: %zu blocks merged (was %zu, now %zu)", 
                compacted, original_count, pool->free_blocks_count);
    }

    return compacted > 0;
}

void zero_copy_allocation_destroy(zero_copy_allocation_t* allocation) {
    if (allocation) {
        // Note: ne libère PAS le pointeur allocation->ptr 
        // car c'est fait par zero_copy_free()
        free(allocation);
    }
}
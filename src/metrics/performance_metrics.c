
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include "performance_metrics.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>

// Create performance metrics context
performance_metrics_t* performance_metrics_create(void) {
    performance_metrics_t* ctx = malloc(sizeof(performance_metrics_t));
    if (!ctx) return NULL;
    
    ctx->metric_count = 0;
    ctx->is_initialized = false;
    ctx->total_operations = 0;
    ctx->cpu_usage = 0.0;
    ctx->memory_usage = 0;
    ctx->peak_memory = 0;
    
    get_current_timespec(&ctx->start_time);
    
    // Initialize all metrics as inactive
    for (size_t i = 0; i < MAX_METRICS_COUNT; i++) {
        ctx->metrics[i].is_active = false;
    }
    
    // Register default system metrics
    performance_metrics_register(ctx, "system.cpu_usage", METRIC_GAUGE);
    performance_metrics_register(ctx, "system.memory_usage", METRIC_GAUGE);
    performance_metrics_register(ctx, "system.operations_total", METRIC_COUNTER);
    performance_metrics_register(ctx, "lum.creation_time", METRIC_HISTOGRAM);
    performance_metrics_register(ctx, "vorax.operation_time", METRIC_HISTOGRAM);
    performance_metrics_register(ctx, "binary.conversion_time", METRIC_HISTOGRAM);
    
    ctx->is_initialized = true;
    return ctx;
}

void performance_metrics_destroy(performance_metrics_t* ctx) {
    if (ctx) {
        free(ctx);
    }
}

// Metric registration
bool performance_metrics_register(performance_metrics_t* ctx, const char* name, metric_type_e type) {
    if (!ctx || !name || !is_metric_name_valid(name)) return false;
    
    if (ctx->metric_count >= MAX_METRICS_COUNT) return false;
    
    // Check if metric already exists
    for (size_t i = 0; i < ctx->metric_count; i++) {
        if (strcmp(ctx->metrics[i].name, name) == 0) {
            return false; // Already exists
        }
    }
    
    performance_metric_t* metric = &ctx->metrics[ctx->metric_count];
    strncpy(metric->name, name, MAX_METRIC_NAME_LENGTH - 1);
    metric->name[MAX_METRIC_NAME_LENGTH - 1] = '\0';
    
    metric->type = type;
    metric->value = 0.0;
    metric->min_value = INFINITY;
    metric->max_value = -INFINITY;
    metric->sum_value = 0.0;
    metric->count = 0;
    get_current_timespec(&metric->last_updated);
    metric->is_active = true;
    
    ctx->metric_count++;
    return true;
}

performance_metric_t* performance_metrics_get(performance_metrics_t* ctx, const char* name) {
    if (!ctx || !name) return NULL;
    
    for (size_t i = 0; i < ctx->metric_count; i++) {
        if (ctx->metrics[i].is_active && strcmp(ctx->metrics[i].name, name) == 0) {
            return &ctx->metrics[i];
        }
    }
    
    return NULL;
}

// Metric operations
bool performance_metrics_increment_counter(performance_metrics_t* ctx, const char* name, double value) {
    performance_metric_t* metric = performance_metrics_get(ctx, name);
    if (!metric || metric->type != METRIC_COUNTER) return false;
    
    metric->value += value;
    metric->sum_value += value;
    metric->count++;
    get_current_timespec(&metric->last_updated);
    
    return true;
}

bool performance_metrics_set_gauge(performance_metrics_t* ctx, const char* name, double value) {
    performance_metric_t* metric = performance_metrics_get(ctx, name);
    if (!metric || metric->type != METRIC_GAUGE) return false;
    
    metric->value = value;
    if (value < metric->min_value) metric->min_value = value;
    if (value > metric->max_value) metric->max_value = value;
    metric->sum_value += value;
    metric->count++;
    get_current_timespec(&metric->last_updated);
    
    return true;
}

bool performance_metrics_record_histogram(performance_metrics_t* ctx, const char* name, double value) {
    performance_metric_t* metric = performance_metrics_get(ctx, name);
    if (!metric || metric->type != METRIC_HISTOGRAM) return false;
    
    if (value < metric->min_value) metric->min_value = value;
    if (value > metric->max_value) metric->max_value = value;
    metric->sum_value += value;
    metric->count++;
    metric->value = metric->sum_value / metric->count; // Running average
    get_current_timespec(&metric->last_updated);
    
    return true;
}

// Timer operations
operation_timer_t* operation_timer_create(void) {
    operation_timer_t* timer = malloc(sizeof(operation_timer_t));
    if (!timer) return NULL;
    
    timer->is_running = false;
    timer->elapsed_seconds = 0.0;
    
    return timer;
}

void operation_timer_destroy(operation_timer_t* timer) {
    if (timer) {
        free(timer);
    }
}

bool operation_timer_start(operation_timer_t* timer) {
    if (!timer || timer->is_running) return false;
    
    get_current_timespec(&timer->start_time);
    timer->is_running = true;
    
    return true;
}

bool operation_timer_stop(operation_timer_t* timer) {
    if (!timer || !timer->is_running) return false;
    
    get_current_timespec(&timer->end_time);
    timer->is_running = false;
    
    timer->elapsed_seconds = timespec_to_seconds(&timer->end_time) - 
                            timespec_to_seconds(&timer->start_time);
    
    return true;
}

double operation_timer_get_elapsed(operation_timer_t* timer) {
    if (!timer) return 0.0;
    
    if (timer->is_running) {
        struct timespec current_time;
        get_current_timespec(&current_time);
        return timespec_to_seconds(&current_time) - timespec_to_seconds(&timer->start_time);
    }
    
    return timer->elapsed_seconds;
}

// System metrics
bool performance_metrics_update_system_stats(performance_metrics_t* ctx) {
    if (!ctx) return false;
    
    // Update CPU usage
    double cpu = performance_metrics_get_cpu_usage();
    performance_metrics_set_gauge(ctx, "system.cpu_usage", cpu);
    ctx->cpu_usage = cpu;
    
    // Update memory usage
    size_t memory = performance_metrics_get_memory_usage();
    performance_metrics_set_gauge(ctx, "system.memory_usage", (double)memory);
    ctx->memory_usage = memory;
    if (memory > ctx->peak_memory) {
        ctx->peak_memory = memory;
    }
    
    // Update operations counter
    ctx->total_operations++;
    performance_metrics_increment_counter(ctx, "system.operations_total", 1.0);
    
    return true;
}

double performance_metrics_get_cpu_usage(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double user_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0;
        double sys_time = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0;
        return user_time + sys_time;
    }
    return 0.0;
}

size_t performance_metrics_get_memory_usage(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss * 1024; // Convert KB to bytes
    }
    return 0;
}

// Reporting
void performance_metrics_print_summary(performance_metrics_t* ctx) {
    if (!ctx) return;
    
    printf("=== Performance Metrics Summary ===\n");
    printf("Uptime: %.2f seconds\n", 
           timespec_to_seconds(&ctx->start_time));
    printf("Total Operations: %lu\n", ctx->total_operations);
    printf("CPU Usage: %.2f%%\n", ctx->cpu_usage * 100.0);
    printf("Memory Usage: %zu bytes\n", ctx->memory_usage);
    printf("Peak Memory: %zu bytes\n", ctx->peak_memory);
    printf("Active Metrics: %zu\n", ctx->metric_count);
    printf("===================================\n");
}

void performance_metrics_print_detailed(performance_metrics_t* ctx) {
    if (!ctx) return;
    
    performance_metrics_print_summary(ctx);
    
    printf("\n=== Detailed Metrics ===\n");
    for (size_t i = 0; i < ctx->metric_count; i++) {
        performance_metric_t* metric = &ctx->metrics[i];
        if (!metric->is_active) continue;
        
        const char* type_str = (metric->type == METRIC_COUNTER) ? "COUNTER" :
                               (metric->type == METRIC_GAUGE) ? "GAUGE" :
                               (metric->type == METRIC_HISTOGRAM) ? "HISTOGRAM" : "TIMER";
        
        printf("%s [%s]: ", metric->name, type_str);
        printf("value=%.6f, count=%lu", metric->value, metric->count);
        
        if (metric->count > 0 && metric->type != METRIC_COUNTER) {
            printf(", min=%.6f, max=%.6f, avg=%.6f", 
                   metric->min_value, metric->max_value, 
                   metric->sum_value / metric->count);
        }
        printf("\n");
    }
    printf("========================\n");
}

// Benchmarking
bool performance_metrics_benchmark_operation(performance_metrics_t* ctx, 
                                            const char* name, 
                                            void (*operation)(void*), 
                                            void* data,
                                            int iterations) {
    if (!ctx || !name || !operation || iterations <= 0) return false;
    
    // Register benchmark metric if not exists
    char timer_name[MAX_METRIC_NAME_LENGTH];
    snprintf(timer_name, sizeof(timer_name), "benchmark.%s", name);
    performance_metrics_register(ctx, timer_name, METRIC_HISTOGRAM);
    
    operation_timer_t* timer = operation_timer_create();
    if (!timer) return false;
    
    for (int i = 0; i < iterations; i++) {
        operation_timer_start(timer);
        operation(data);
        operation_timer_stop(timer);
        
        double elapsed = operation_timer_get_elapsed(timer);
        performance_metrics_record_histogram(ctx, timer_name, elapsed);
    }
    
    operation_timer_destroy(timer);
    return true;
}

// Utility functions
double timespec_to_seconds(struct timespec* ts) {
    if (!ts) return 0.0;
    return ts->tv_sec + ts->tv_nsec / 1000000000.0;
}

void get_current_timespec(struct timespec* ts) {
    if (ts) {
        clock_gettime(CLOCK_MONOTONIC, ts);
    }
}

bool is_metric_name_valid(const char* name) {
    if (!name || strlen(name) == 0 || strlen(name) >= MAX_METRIC_NAME_LENGTH) {
        return false;
    }
    
    // Check for valid characters (alphanumeric, dots, underscores)
    for (const char* p = name; *p; p++) {
        if (!(*p >= 'a' && *p <= 'z') && 
            !(*p >= 'A' && *p <= 'Z') && 
            !(*p >= '0' && *p <= '9') && 
            *p != '.' && *p != '_') {
            return false;
        }
    }
    
    return true;
}

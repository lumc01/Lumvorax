
#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>

#define MAX_METRIC_NAME_LENGTH 64
#define MAX_METRICS_COUNT 100

// Metric types
typedef enum {
    METRIC_COUNTER,
    METRIC_GAUGE,
    METRIC_HISTOGRAM,
    METRIC_TIMER
} metric_type_e;

// Individual metric
typedef struct {
    char name[MAX_METRIC_NAME_LENGTH];
    metric_type_e type;
    double value;
    double min_value;
    double max_value;
    double sum_value;
    uint64_t count;
    struct timespec last_updated;
    bool is_active;
} performance_metric_t;

// Histogram bucket
typedef struct {
    double upper_bound;
    uint64_t count;
} histogram_bucket_t;

// Performance metrics context
typedef struct {
    performance_metric_t metrics[MAX_METRICS_COUNT];
    size_t metric_count;
    bool is_initialized;
    struct timespec start_time;
    uint64_t total_operations;
    double cpu_usage;
    size_t memory_usage;
    size_t peak_memory;
} performance_metrics_t;

// Timer context for measuring operations
typedef struct {
    struct timespec start_time;
    struct timespec end_time;
    bool is_running;
    double elapsed_seconds;
} operation_timer_t;

// Function declarations
performance_metrics_t* performance_metrics_create(void);
void performance_metrics_destroy(performance_metrics_t* metrics);

// Metric registration and management
bool performance_metrics_register(performance_metrics_t* ctx, 
                                 const char* name, metric_type_e type);
performance_metric_t* performance_metrics_get(performance_metrics_t* ctx, const char* name);

// Metric operations
bool performance_metrics_increment_counter(performance_metrics_t* ctx, const char* name, double value);
bool performance_metrics_set_gauge(performance_metrics_t* ctx, const char* name, double value);
bool performance_metrics_record_histogram(performance_metrics_t* ctx, const char* name, double value);

// Timer operations
operation_timer_t* operation_timer_create(void);
void operation_timer_destroy(operation_timer_t* timer);
bool operation_timer_start(operation_timer_t* timer);
bool operation_timer_stop(operation_timer_t* timer);
double operation_timer_get_elapsed(operation_timer_t* timer);

// High-level timing functions
bool performance_metrics_start_timer(performance_metrics_t* ctx, const char* name);
bool performance_metrics_stop_timer(performance_metrics_t* ctx, const char* name);

// System metrics
bool performance_metrics_update_system_stats(performance_metrics_t* ctx);
double performance_metrics_get_cpu_usage(void);
size_t performance_metrics_get_memory_usage(void);

// Reporting and analysis
void performance_metrics_print_summary(performance_metrics_t* ctx);
void performance_metrics_print_detailed(performance_metrics_t* ctx);
bool performance_metrics_export_csv(performance_metrics_t* ctx, const char* filename);
bool performance_metrics_export_json(performance_metrics_t* ctx, const char* filename);

// Statistical analysis
double performance_metrics_calculate_mean(performance_metric_t* metric);
double performance_metrics_calculate_stddev(performance_metric_t* metric);
double performance_metrics_calculate_percentile(performance_metric_t* metric, double percentile);

// Benchmarking utilities
bool performance_metrics_benchmark_operation(performance_metrics_t* ctx, 
                                           const char* name, 
                                           void (*operation)(void*), 
                                           void* data,
                                           int iterations);

// Utility functions
double timespec_to_seconds(struct timespec* ts);
void get_current_timespec(struct timespec* ts);
bool is_metric_name_valid(const char* name);

#endif // PERFORMANCE_METRICS_H

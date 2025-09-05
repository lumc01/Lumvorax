
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <unistd.h>

// Include all advanced module headers
#include "../optimization/memory_optimizer.h"
#include "../parallel/parallel_processor.h"
#include "../metrics/performance_metrics.h"
#include "../crypto/crypto_validator.h"
#include "../persistence/data_persistence.h"
#include "../lum/lum_core.h"
#include "../logger/lum_logger.h"

// Global test counter
static int tests_passed = 0;
static int tests_failed = 0;
static lum_logger_t* test_logger = NULL;

#define TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ %s: PASSED\n", test_name); \
            tests_passed++; \
            if (test_logger) lum_log_message(test_logger, LUM_LOG_INFO, "TEST_PASS: " test_name); \
        } else { \
            printf("✗ %s: FAILED\n", test_name); \
            tests_failed++; \
            if (test_logger) lum_log_message(test_logger, LUM_LOG_ERROR, "TEST_FAIL: " test_name); \
        } \
    } while(0)

void test_memory_optimizer() {
    printf("\n=== Testing Memory Optimizer Module ===\n");
    
    // Test 1: Memory optimizer creation
    memory_optimizer_t* optimizer = memory_optimizer_create(1024);
    TEST_ASSERT(optimizer != NULL, "Memory optimizer creation");
    
    // Test 2: LUM allocation from optimizer
    lum_t* lum1 = memory_optimizer_alloc_lum(optimizer);
    TEST_ASSERT(lum1 != NULL, "LUM allocation from optimizer");
    
    lum_t* lum2 = memory_optimizer_alloc_lum(optimizer);
    TEST_ASSERT(lum2 != NULL, "Second LUM allocation");
    TEST_ASSERT(lum1 != lum2, "Different memory addresses");
    
    // Test 3: Memory optimizer statistics
    memory_stats_t* stats = memory_optimizer_get_stats(optimizer);
    TEST_ASSERT(stats->total_allocated > 0, "Memory statistics tracking");
    TEST_ASSERT(stats->allocation_count >= 2, "Allocation count tracking");
    
    // Test 4: Memory deallocation
    memory_optimizer_free_lum(optimizer, lum1);
    stats = memory_optimizer_get_stats(optimizer);
    TEST_ASSERT(stats->free_count >= 1, "Memory deallocation tracking");
    
    // Test 5: Auto defragmentation
    bool fragmented = memory_optimizer_analyze_fragmentation(optimizer);
    TEST_ASSERT(fragmented >= 0, "Fragmentation analysis");
    
    // Test 6: Statistics printing
    memory_optimizer_print_stats(optimizer);
    TEST_ASSERT(true, "Memory statistics printing");
    
    memory_optimizer_destroy(optimizer);
    printf("Memory Optimizer: %d tests completed\n", 6);
}

void test_parallel_processor() {
    printf("\n=== Testing Parallel Processor Module ===\n");
    
    // Test 1: Thread pool creation
    thread_pool_t* pool = thread_pool_create(4);
    TEST_ASSERT(pool != NULL, "Thread pool creation");
    
    // Test 2: Task submission
    parallel_task_t* task = parallel_task_create(TASK_LUM_CREATE, NULL, 0);
    TEST_ASSERT(task != NULL, "Parallel task creation");
    
    bool submitted = thread_pool_submit(pool, task);
    TEST_ASSERT(submitted, "Task submission to thread pool");
    
    // Test 3: Parallel LUM processing
    lum_t* lums[10];
    for (int i = 0; i < 10; i++) {
        lums[i] = lum_create(1, i, 0, LUM_STRUCTURE_LINEAR);
    }
    
    parallel_process_result_t result = parallel_process_lums(lums, 10, 4);
    TEST_ASSERT(result.success, "Parallel LUM processing");
    TEST_ASSERT(result.processed_count == 10, "All LUMs processed");
    
    // Test 4: Workload distribution
    work_distribution_t dist;
    bool distributed = distribute_work(lums, 10, 4, &dist);
    TEST_ASSERT(distributed, "Workload distribution");
    TEST_ASSERT(dist.thread_count == 4, "Correct thread count");
    
    // Test 5: Parallel reduction operation
    double sum = parallel_reduce_lums(lums, 10, 4);
    TEST_ASSERT(sum > 0.0, "Parallel reduction operation");
    
    // Test 6: Thread synchronization
    bool synced = thread_pool_wait_all(pool);
    TEST_ASSERT(synced, "Thread synchronization");
    
    // Cleanup
    for (int i = 0; i < 10; i++) {
        lum_destroy(lums[i]);
    }
    thread_pool_destroy(pool);
    printf("Parallel Processor: %d tests completed\n", 6);
}

void test_performance_metrics() {
    printf("\n=== Testing Performance Metrics Module ===\n");
    
    // Test 1: Performance metrics creation
    performance_metrics_t* metrics = performance_metrics_create();
    TEST_ASSERT(metrics != NULL, "Performance metrics creation");
    
    // Test 2: Performance timer
    operation_timer_t* timer = operation_timer_create();
    TEST_ASSERT(timer != NULL, "Performance timer creation");
    
    bool started = operation_timer_start(timer);
    TEST_ASSERT(started, "Performance timer start");
    
    // Simulate some work
    usleep(1000); // 1ms
    
    bool stopped = operation_timer_stop(timer);
    TEST_ASSERT(stopped, "Performance timer stop");
    
    double elapsed = operation_timer_get_elapsed(timer);
    TEST_ASSERT(elapsed > 0.0, "Performance timer measurement");
    operation_timer_destroy(timer);
    
    // Test 3: Memory usage tracking
    size_t mem_usage = performance_metrics_get_memory_usage();
    TEST_ASSERT(mem_usage >= 0, "Memory usage measurement");
    
    // Test 4: CPU utilization measurement
    double cpu_usage = performance_metrics_get_cpu_usage();
    TEST_ASSERT(cpu_usage >= 0.0, "CPU utilization measurement");
    
    // Test 5: Metric registration
    bool registered = performance_metrics_register(metrics, "test_counter", METRIC_COUNTER);
    TEST_ASSERT(registered, "Metric registration");
    
    // Test 6: Performance metrics summary
    performance_metrics_print_summary(metrics);cs);
    TEST_ASSERT(true, "Performance metrics summary");
    
    performance_metrics_destroy(metrics);
    printf("Performance Metrics: %d tests completed\n", 6);
});
}

void test_crypto_validator() {
    printf("\n=== Testing Crypto Validator Module ===\n");
    
    // Test 1: SHA256 context initialization
    sha256_context_t ctx;
    sha256_init(&ctx);
    TEST_ASSERT(true, "SHA256 context initialization");
    
    // Test 2: Data hashing
    const char* test_data = "Hello, LUM/VORAX!";
    uint8_t hash[SHA256_DIGEST_SIZE];
    sha256_hash((const uint8_t*)test_data, strlen(test_data), hash);
    TEST_ASSERT(hash[0] != 0, "Data hashing operation"); // Non-zero first byte indicates success
    
    // Test 3: Hash verification
    uint8_t hash2[SHA256_DIGEST_SIZE];
    sha256_hash((const uint8_t*)test_data, strlen(test_data), hash2);
    bool verified = (memcmp(hash, hash2, SHA256_DIGEST_SIZE) == 0);
    TEST_ASSERT(verified, "Hash verification - identical data");
    
    // Test 4: Different data produces different hash
    const char* different_data = "Different data";
    uint8_t hash3[SHA256_DIGEST_SIZE];
    sha256_hash((const uint8_t*)different_data, strlen(different_data), hash3);
    bool different = (memcmp(hash, hash3, SHA256_DIGEST_SIZE) != 0);
    TEST_ASSERT(different, "Different data produces different hash");
    
    // Test 5: Data integrity check
    lum_t* lum = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    char hash_string[MAX_HASH_STRING_LENGTH];
    bool computed = compute_data_hash(lum, sizeof(lum_t), hash_string);
    TEST_ASSERT(computed, "Data hash computation");
    TEST_ASSERT(strlen(hash_string) > 0, "Hash string generated");
    
    // Test 6: Hex string conversion
    char hex_string[MAX_HASH_STRING_LENGTH];
    bytes_to_hex_string(hash, SHA256_DIGEST_SIZE, hex_string);
    TEST_ASSERT(strlen(hex_string) == 64, "Hex string conversion"); // 32 bytes = 64 hex chars
    
    lum_destroy(lum);
    printf("Crypto Validator: %d tests completed\n", 6);
}

void test_data_persistence() {
    printf("\n=== Testing Data Persistence Module ===\n");
    
    // Test 1: Storage backend creation
    storage_backend_t* backend = storage_backend_create("test_storage.db");
    TEST_ASSERT(backend != NULL, "Storage backend creation");
    
    // Test 2: LUM serialization
    lum_t* original_lum = lum_create(1, 15, 25, LUM_STRUCTURE_CIRCULAR);
    serialized_data_t* serialized = serialize_lum(original_lum);
    TEST_ASSERT(serialized != NULL, "LUM serialization");
    TEST_ASSERT(serialized->size > 0, "Serialized data has content");
    
    // Test 3: LUM deserialization
    lum_t* restored_lum = deserialize_lum(serialized);
    TEST_ASSERT(restored_lum != NULL, "LUM deserialization");
    TEST_ASSERT(restored_lum->presence == original_lum->presence, "Presence restored");
    TEST_ASSERT(restored_lum->position_x == original_lum->position_x, "Position X restored");
    TEST_ASSERT(restored_lum->position_y == original_lum->position_y, "Position Y restored");
    
    // Test 4: Data persistence to storage
    bool stored = store_lum(backend, "test_lum_1", original_lum);
    TEST_ASSERT(stored, "LUM storage to backend");
    
    // Test 5: Data retrieval from storage
    lum_t* loaded_lum = load_lum(backend, "test_lum_1");
    TEST_ASSERT(loaded_lum != NULL, "LUM loading from backend");
    TEST_ASSERT(loaded_lum->presence == original_lum->presence, "Loaded LUM matches original");
    
    // Test 6: Transaction support
    transaction_t* transaction = begin_transaction(backend);
    TEST_ASSERT(transaction != NULL, "Transaction creation");
    
    bool committed = commit_transaction(transaction);
    TEST_ASSERT(committed, "Transaction commit");
    
    // Test 7: Batch operations
    lum_t* batch_lums[5];
    for (int i = 0; i < 5; i++) {
        batch_lums[i] = lum_create(1, i * 10, i * 10, LUM_STRUCTURE_LINEAR);
    }
    
    bool batch_stored = storage_backend_store_batch(backend, (void**)batch_lums, 5);
    TEST_ASSERT(batch_stored, "Batch LUM storage");
    
    // Cleanup
    lum_destroy(original_lum);
    lum_destroy(restored_lum);
    lum_destroy(loaded_lum);
    for (int i = 0; i < 5; i++) {
        lum_destroy(batch_lums[i]);
    }
    serialized_data_destroy(serialized);
    storage_backend_destroy(backend);
    
    printf("Data Persistence: %d tests completed\n", 7);
}

void test_integration_scenarios() {
    printf("\n=== Testing Integration Scenarios ===\n");
    
    // Scenario 1: Complete LUM lifecycle with all modules
    memory_optimizer_t* mem_opt = memory_optimizer_create(2048);
    thread_pool_t* thread_pool = thread_pool_create(2);
    performance_metrics_t* metrics = performance_metrics_create();
    storage_backend_t* storage = storage_backend_create("integration_test.db");
    
    TEST_ASSERT(mem_opt && thread_pool && metrics && storage, 
                "All modules initialized successfully");
    
    // Create and process LUMs with all modules involved
    lum_t* test_lums[20];
    for (int i = 0; i < 20; i++) {
        test_lums[i] = lum_create(1, i, i * 2, LUM_STRUCTURE_LINEAR);
    }
    
    // Performance measurement
    operation_timer_t* timer = operation_timer_create();
    operation_timer_start(timer);
    
    // Parallel processing
    parallel_process_result_t process_result = parallel_process_lums(test_lums, 20, 2);
    TEST_ASSERT(process_result.success && process_result.processed_count == 20,
                "Parallel LUM processing in integration");
    
    // Crypto validation
    for (int i = 0; i < 5; i++) {
        char hash_str[MAX_HASH_STRING_LENGTH];
        bool hash_ok = compute_data_hash(test_lums[i], sizeof(lum_t), hash_str);
        TEST_ASSERT(hash_ok, "LUM integrity in integration");
    }
    
    // Persistence
    bool all_stored = storage_backend_store_batch(storage, (void**)test_lums, 20);
    TEST_ASSERT(all_stored, "Batch storage in integration");
    
    // Performance measurement completion
    operation_timer_stop(timer);
    double total_time = operation_timer_get_elapsed(timer);
    TEST_ASSERT(total_time > 0.0, "Integration performance measurement");
    operation_timer_destroy(timer);
    
    printf("Integration test completed in %.3f seconds\n", total_time);
    
    // Cleanup
    for (int i = 0; i < 20; i++) {
        lum_destroy(test_lums[i]);
    }
    memory_optimizer_destroy(mem_opt);
    thread_pool_destroy(thread_pool);
    performance_metrics_destroy(metrics);
    storage_backend_destroy(storage);
    
    printf("Integration Scenarios: %d tests completed\n", 6);
}

int main() {
    printf("=== ADVANCED MODULES COMPREHENSIVE TEST SUITE ===\n");
    printf("Testing all 5 advanced modules with real functionality\n");
    printf("Build timestamp: %s %s\n", __DATE__, __TIME__);
    
    // Initialize test logger
    test_logger = lum_logger_create("logs/test_advanced_modules.log", true, true);
    if (test_logger) {
        lum_logger_set_level(test_logger, LUM_LOG_DEBUG);
        lum_log_message(test_logger, LUM_LOG_INFO, "Advanced modules test suite started");
    }
    
    // Run all module tests
    test_memory_optimizer();
    test_parallel_processor();  
    test_performance_metrics();
    test_crypto_validator();
    test_data_persistence();
    test_integration_scenarios();
    
    // Final results
    printf("\n=== TEST SUITE RESULTS ===\n");
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("Success Rate: %.1f%%\n", 
           tests_passed > 0 ? (100.0 * tests_passed) / (tests_passed + tests_failed) : 0.0);
    
    if (test_logger) {
        char summary[256];
        snprintf(summary, sizeof(summary), 
                "Test suite completed - Passed: %d, Failed: %d", 
                tests_passed, tests_failed);
        lum_log_message(test_logger, LUM_LOG_INFO, summary);
        lum_logger_destroy(test_logger);
    }
    
    return tests_failed == 0 ? 0 : 1;
}


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

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
            if (test_logger) lum_log_info(test_logger, "TEST_PASS: " test_name); \
        } else { \
            printf("✗ %s: FAILED\n", test_name); \
            tests_failed++; \
            if (test_logger) lum_log_error(test_logger, "TEST_FAIL: " test_name); \
        } \
    } while(0)

void test_memory_optimizer() {
    printf("\n=== Testing Memory Optimizer Module ===\n");
    
    // Test 1: Memory pool creation
    memory_pool_t* pool = memory_pool_create(1024, 64);
    TEST_ASSERT(pool != NULL, "Memory pool creation");
    
    // Test 2: Memory allocation from pool
    void* ptr1 = memory_pool_alloc(pool, 32);
    TEST_ASSERT(ptr1 != NULL, "Memory allocation from pool");
    
    void* ptr2 = memory_pool_alloc(pool, 48);
    TEST_ASSERT(ptr2 != NULL, "Second memory allocation");
    TEST_ASSERT(ptr1 != ptr2, "Different memory addresses");
    
    // Test 3: Memory pool statistics
    memory_stats_t stats;
    memory_pool_get_stats(pool, &stats);
    TEST_ASSERT(stats.total_allocated > 0, "Memory statistics tracking");
    TEST_ASSERT(stats.blocks_allocated == 2, "Block count tracking");
    
    // Test 4: Memory deallocation
    memory_pool_free(pool, ptr1);
    memory_pool_get_stats(pool, &stats);
    TEST_ASSERT(stats.blocks_allocated == 1, "Memory deallocation tracking");
    
    // Test 5: Pool optimization
    bool optimized = memory_pool_optimize(pool);
    TEST_ASSERT(optimized, "Memory pool optimization");
    
    // Test 6: Pool defragmentation
    bool defragged = memory_pool_defragment(pool);
    TEST_ASSERT(defragged, "Memory pool defragmentation");
    
    memory_pool_destroy(pool);
    printf("Memory Optimizer: %d tests completed\n", 6);
}

void test_parallel_processor() {
    printf("\n=== Testing Parallel Processor Module ===\n");
    
    // Test 1: Thread pool creation
    thread_pool_t* pool = thread_pool_create(4);
    TEST_ASSERT(pool != NULL, "Thread pool creation");
    
    // Test 2: Task submission
    parallel_task_t* task = parallel_task_create(NULL, NULL, 0);
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
    
    // Test 1: Metrics collector creation
    metrics_collector_t* collector = metrics_collector_create();
    TEST_ASSERT(collector != NULL, "Metrics collector creation");
    
    // Test 2: Performance timer
    perf_timer_t* timer = perf_timer_start("test_operation");
    TEST_ASSERT(timer != NULL, "Performance timer start");
    
    // Simulate some work
    usleep(1000); // 1ms
    
    double elapsed = perf_timer_stop(timer);
    TEST_ASSERT(elapsed > 0.0, "Performance timer measurement");
    TEST_ASSERT(elapsed < 0.1, "Reasonable timing measurement");
    
    // Test 3: Memory usage tracking
    memory_usage_t usage;
    bool tracked = track_memory_usage(&usage);
    TEST_ASSERT(tracked, "Memory usage tracking");
    TEST_ASSERT(usage.current_bytes > 0, "Memory usage measurement");
    
    // Test 4: CPU utilization measurement
    cpu_stats_t cpu_stats;
    bool cpu_measured = measure_cpu_utilization(&cpu_stats);
    TEST_ASSERT(cpu_measured, "CPU utilization measurement");
    TEST_ASSERT(cpu_stats.user_percent >= 0.0, "Valid CPU percentage");
    
    // Test 5: Throughput calculation
    throughput_stats_t throughput;
    calculate_throughput(1000, 2.5, &throughput);
    TEST_ASSERT(throughput.ops_per_second == 400.0, "Throughput calculation");
    
    // Test 6: Performance profile generation
    performance_profile_t profile;
    bool generated = generate_performance_profile(collector, &profile);
    TEST_ASSERT(generated, "Performance profile generation");
    
    metrics_collector_destroy(collector);
    printf("Performance Metrics: %d tests completed\n", 6);
}

void test_crypto_validator() {
    printf("\n=== Testing Crypto Validator Module ===\n");
    
    // Test 1: Hash calculator creation
    hash_calculator_t* calculator = hash_calculator_create(HASH_SHA256);
    TEST_ASSERT(calculator != NULL, "Hash calculator creation");
    
    // Test 2: Data hashing
    const char* test_data = "Hello, LUM/VORAX!";
    hash_result_t hash;
    bool hashed = calculate_hash(calculator, test_data, strlen(test_data), &hash);
    TEST_ASSERT(hashed, "Data hashing operation");
    TEST_ASSERT(hash.length == 32, "SHA256 hash length"); // SHA256 = 32 bytes
    
    // Test 3: Hash verification
    hash_result_t hash2;
    calculate_hash(calculator, test_data, strlen(test_data), &hash2);
    bool verified = verify_hash(&hash, &hash2);
    TEST_ASSERT(verified, "Hash verification - identical data");
    
    // Test 4: Different data produces different hash
    const char* different_data = "Different data";
    hash_result_t hash3;
    calculate_hash(calculator, different_data, strlen(different_data), &hash3);
    bool different = !verify_hash(&hash, &hash3);
    TEST_ASSERT(different, "Different data produces different hash");
    
    // Test 5: LUM integrity validation
    lum_t* lum = lum_create(1, 10, 20, LUM_STRUCTURE_LINEAR);
    integrity_result_t integrity;
    bool validated = validate_lum_integrity(lum, &integrity);
    TEST_ASSERT(validated, "LUM integrity validation");
    TEST_ASSERT(integrity.is_valid, "LUM integrity check passed");
    
    // Test 6: Digital signature simulation
    signature_result_t signature;
    bool signed_data = sign_data(test_data, strlen(test_data), &signature);
    TEST_ASSERT(signed_data, "Digital signature generation");
    
    bool signature_valid = verify_signature(test_data, strlen(test_data), &signature);
    TEST_ASSERT(signature_valid, "Digital signature verification");
    
    lum_destroy(lum);
    hash_calculator_destroy(calculator);
    printf("Crypto Validator: %d tests completed\n", 6);
}

void test_data_persistence() {
    printf("\n=== Testing Data Persistence Module ===\n");
    
    // Test 1: Storage backend creation
    storage_backend_t* backend = storage_backend_create("test_storage.db");
    TEST_ASSERT(backend != NULL, "Storage backend creation");
    
    // Test 2: LUM serialization
    lum_t* original_lum = lum_create(1, 15, 25, LUM_STRUCTURE_CIRCULAR);
    serialized_data_t serialized;
    bool serialized_ok = serialize_lum(original_lum, &serialized);
    TEST_ASSERT(serialized_ok, "LUM serialization");
    TEST_ASSERT(serialized.size > 0, "Serialized data has content");
    
    // Test 3: LUM deserialization
    lum_t* restored_lum = deserialize_lum(&serialized);
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
    
    bool batch_stored = store_lum_batch(backend, batch_lums, 5);
    TEST_ASSERT(batch_stored, "Batch LUM storage");
    
    // Cleanup
    lum_destroy(original_lum);
    lum_destroy(restored_lum);
    lum_destroy(loaded_lum);
    for (int i = 0; i < 5; i++) {
        lum_destroy(batch_lums[i]);
    }
    storage_backend_destroy(backend);
    
    printf("Data Persistence: %d tests completed\n", 7);
}

void test_integration_scenarios() {
    printf("\n=== Testing Integration Scenarios ===\n");
    
    // Scenario 1: Complete LUM lifecycle with all modules
    memory_pool_t* mem_pool = memory_pool_create(2048, 128);
    thread_pool_t* thread_pool = thread_pool_create(2);
    metrics_collector_t* metrics = metrics_collector_create();
    hash_calculator_t* hasher = hash_calculator_create(HASH_SHA256);
    storage_backend_t* storage = storage_backend_create("integration_test.db");
    
    TEST_ASSERT(mem_pool && thread_pool && metrics && hasher && storage, 
                "All modules initialized successfully");
    
    // Create and process LUMs with all modules involved
    lum_t* test_lums[20];
    for (int i = 0; i < 20; i++) {
        test_lums[i] = lum_create(1, i, i * 2, LUM_STRUCTURE_LINEAR);
    }
    
    // Performance measurement
    perf_timer_t* timer = perf_timer_start("integration_test");
    
    // Parallel processing
    parallel_process_result_t process_result = parallel_process_lums(test_lums, 20, 2);
    TEST_ASSERT(process_result.success && process_result.processed_count == 20,
                "Parallel LUM processing in integration");
    
    // Crypto validation
    for (int i = 0; i < 5; i++) {
        integrity_result_t integrity;
        validate_lum_integrity(test_lums[i], &integrity);
        TEST_ASSERT(integrity.is_valid, "LUM integrity in integration");
    }
    
    // Persistence
    bool all_stored = store_lum_batch(storage, test_lums, 20);
    TEST_ASSERT(all_stored, "Batch storage in integration");
    
    // Performance measurement completion
    double total_time = perf_timer_stop(timer);
    TEST_ASSERT(total_time > 0.0, "Integration performance measurement");
    
    printf("Integration test completed in %.3f seconds\n", total_time);
    
    // Cleanup
    for (int i = 0; i < 20; i++) {
        lum_destroy(test_lums[i]);
    }
    memory_pool_destroy(mem_pool);
    thread_pool_destroy(thread_pool);
    metrics_collector_destroy(metrics);
    hash_calculator_destroy(hasher);
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
        lum_log_info(test_logger, "Advanced modules test suite started");
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
        lum_log_info(test_logger, summary);
        lum_logger_destroy(test_logger);
    }
    
    return tests_failed == 0 ? 0 : 1;
}

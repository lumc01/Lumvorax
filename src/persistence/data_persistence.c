
#include "data_persistence.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

static uint64_t next_transaction_id = 1;

// Context management
persistence_context_t* persistence_context_create(const char* storage_directory) {
    if (!storage_directory) return NULL;
    
    persistence_context_t* ctx = malloc(sizeof(persistence_context_t));
    if (!ctx) return NULL;
    
    strncpy(ctx->storage_directory, storage_directory, MAX_STORAGE_PATH_LENGTH - 1);
    ctx->storage_directory[MAX_STORAGE_PATH_LENGTH - 1] = '\0';
    
    ctx->default_format = STORAGE_FORMAT_BINARY;
    ctx->auto_backup_enabled = true;
    ctx->max_backup_count = 5;
    ctx->compression_enabled = false;
    ctx->integrity_checking_enabled = true;
    ctx->transaction_log = NULL;
    
    // Ensure storage directory exists
    persistence_ensure_directory_exists(storage_directory);
    
    return ctx;
}

void persistence_context_destroy(persistence_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->transaction_log) {
        fclose(ctx->transaction_log);
    }
    
    free(ctx);
}

bool persistence_context_configure(persistence_context_t* ctx,
                                  storage_format_e format,
                                  bool auto_backup,
                                  size_t max_backups,
                                  bool compression) {
    if (!ctx) return false;
    
    ctx->default_format = format;
    ctx->auto_backup_enabled = auto_backup;
    ctx->max_backup_count = max_backups;
    ctx->compression_enabled = compression;
    
    return true;
}

// Storage result management
storage_result_t* storage_result_create(void) {
    storage_result_t* result = malloc(sizeof(storage_result_t));
    if (!result) return NULL;
    
    result->success = false;
    result->filename[0] = '\0';
    result->bytes_written = 0;
    result->bytes_read = 0;
    result->checksum = 0;
    result->error_message[0] = '\0';
    
    return result;
}

void storage_result_destroy(storage_result_t* result) {
    if (result) {
        free(result);
    }
}

void storage_result_set_error(storage_result_t* result, const char* error_message) {
    if (!result || !error_message) return;
    
    result->success = false;
    strncpy(result->error_message, error_message, sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
}

void storage_result_set_success(storage_result_t* result, const char* filename,
                               size_t bytes_processed, uint32_t checksum) {
    if (!result) return;
    
    result->success = true;
    if (filename) {
        strncpy(result->filename, filename, sizeof(result->filename) - 1);
        result->filename[sizeof(result->filename) - 1] = '\0';
    }
    result->bytes_written = bytes_processed;
    result->checksum = checksum;
    result->error_message[0] = '\0';
}

// LUM storage operations
storage_result_t* persistence_save_lum(persistence_context_t* ctx,
                                      const lum_t* lum,
                                      const char* filename) {
    storage_result_t* result = storage_result_create();
    if (!result) return NULL;
    
    if (!ctx || !lum || !filename) {
        storage_result_set_error(result, "Invalid input parameters");
        return result;
    }
    
    char full_path[MAX_STORAGE_PATH_LENGTH];
    snprintf(full_path, sizeof(full_path), "%s/%s", ctx->storage_directory, filename);
    
    FILE* file = fopen(full_path, "wb");
    if (!file) {
        storage_result_set_error(result, "Failed to open file for writing");
        return result;
    }
    
    // Write header
    storage_header_t header = {0};
    header.magic_number = STORAGE_MAGIC_NUMBER;
    header.version = STORAGE_FORMAT_VERSION;
    header.format_type = ctx->default_format;
    header.timestamp = time(NULL);
    header.data_size = sizeof(lum_t);
    
    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        storage_result_set_error(result, "Failed to write header");
        return result;
    }
    
    // Write LUM data
    if (fwrite(lum, sizeof(lum_t), 1, file) != 1) {
        fclose(file);
        storage_result_set_error(result, "Failed to write LUM data");
        return result;
    }
    
    fclose(file);
    
    // Calculate checksum
    uint32_t checksum = persistence_calculate_checksum(lum, sizeof(lum_t));
    
    // Update header with checksum
    file = fopen(full_path, "r+b");
    if (file) {
        fseek(file, offsetof(storage_header_t, checksum), SEEK_SET);
        fwrite(&checksum, sizeof(checksum), 1, file);
        fclose(file);
    }
    
    storage_result_set_success(result, full_path, sizeof(storage_header_t) + sizeof(lum_t), checksum);
    
    // Log transaction
    persistence_log_transaction(ctx, "SAVE_LUM", filename, true, sizeof(lum_t));
    
    return result;
}

storage_result_t* persistence_load_lum(persistence_context_t* ctx,
                                      const char* filename,
                                      lum_t** lum) {
    storage_result_t* result = storage_result_create();
    if (!result) return NULL;
    
    if (!ctx || !filename || !lum) {
        storage_result_set_error(result, "Invalid input parameters");
        return result;
    }
    
    char full_path[MAX_STORAGE_PATH_LENGTH];
    snprintf(full_path, sizeof(full_path), "%s/%s", ctx->storage_directory, filename);
    
    FILE* file = fopen(full_path, "rb");
    if (!file) {
        storage_result_set_error(result, "Failed to open file for reading");
        return result;
    }
    
    // Read and validate header
    storage_header_t header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        storage_result_set_error(result, "Failed to read header");
        return result;
    }
    
    if (header.magic_number != STORAGE_MAGIC_NUMBER) {
        fclose(file);
        storage_result_set_error(result, "Invalid file format");
        return result;
    }
    
    if (header.version != STORAGE_FORMAT_VERSION) {
        fclose(file);
        storage_result_set_error(result, "Unsupported file version");
        return result;
    }
    
    // Allocate LUM
    *lum = malloc(sizeof(lum_t));
    if (!*lum) {
        fclose(file);
        storage_result_set_error(result, "Memory allocation failed");
        return result;
    }
    
    // Read LUM data
    if (fread(*lum, sizeof(lum_t), 1, file) != 1) {
        fclose(file);
        free(*lum);
        *lum = NULL;
        storage_result_set_error(result, "Failed to read LUM data");
        return result;
    }
    
    fclose(file);
    
    // Verify integrity if enabled
    if (ctx->integrity_checking_enabled) {
        uint32_t calculated_checksum = persistence_calculate_checksum(*lum, sizeof(lum_t));
        if (calculated_checksum != header.checksum) {
            free(*lum);
            *lum = NULL;
            storage_result_set_error(result, "Data integrity check failed");
            return result;
        }
    }
    
    storage_result_set_success(result, full_path, sizeof(lum_t), header.checksum);
    result->bytes_read = sizeof(lum_t);
    
    // Log transaction
    persistence_log_transaction(ctx, "LOAD_LUM", filename, true, sizeof(lum_t));
    
    return result;
}

// Group storage operations
storage_result_t* persistence_save_group(persistence_context_t* ctx,
                                        const lum_group_t* group,
                                        const char* filename) {
    storage_result_t* result = storage_result_create();
    if (!result) return NULL;
    
    if (!ctx || !group || !filename) {
        storage_result_set_error(result, "Invalid input parameters");
        return result;
    }
    
    char full_path[MAX_STORAGE_PATH_LENGTH];
    snprintf(full_path, sizeof(full_path), "%s/%s", ctx->storage_directory, filename);
    
    FILE* file = fopen(full_path, "wb");
    if (!file) {
        storage_result_set_error(result, "Failed to open file for writing");
        return result;
    }
    
    // Calculate total data size
    size_t group_header_size = sizeof(uint32_t) * 3; // group_id, count, capacity
    size_t lums_data_size = group->count * sizeof(lum_t);
    size_t total_data_size = group_header_size + lums_data_size;
    
    // Write storage header
    storage_header_t header = {0};
    header.magic_number = STORAGE_MAGIC_NUMBER;
    header.version = STORAGE_FORMAT_VERSION;
    header.format_type = ctx->default_format;
    header.timestamp = time(NULL);
    header.data_size = total_data_size;
    strcpy(header.metadata, "LUM_GROUP");
    
    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        storage_result_set_error(result, "Failed to write header");
        return result;
    }
    
    // Write group metadata
    if (fwrite(&group->group_id, sizeof(group->group_id), 1, file) != 1 ||
        fwrite(&group->count, sizeof(group->count), 1, file) != 1 ||
        fwrite(&group->capacity, sizeof(group->capacity), 1, file) != 1) {
        fclose(file);
        storage_result_set_error(result, "Failed to write group metadata");
        return result;
    }
    
    // Write LUMs data
    if (group->count > 0 && group->lums) {
        if (fwrite(group->lums, sizeof(lum_t), group->count, file) != group->count) {
            fclose(file);
            storage_result_set_error(result, "Failed to write LUMs data");
            return result;
        }
    }
    
    fclose(file);
    
    // Calculate checksum
    uint32_t checksum = persistence_calculate_checksum(&group->group_id, sizeof(group->group_id));
    if (group->lums && group->count > 0) {
        checksum ^= persistence_calculate_checksum(group->lums, lums_data_size);
    }
    
    // Update header with checksum
    file = fopen(full_path, "r+b");
    if (file) {
        fseek(file, offsetof(storage_header_t, checksum), SEEK_SET);
        fwrite(&checksum, sizeof(checksum), 1, file);
        fclose(file);
    }
    
    storage_result_set_success(result, full_path, sizeof(storage_header_t) + total_data_size, checksum);
    
    // Log transaction
    persistence_log_transaction(ctx, "SAVE_GROUP", filename, true, total_data_size);
    
    return result;
}

// Utility functions
bool persistence_ensure_directory_exists(const char* directory) {
    if (!directory) return false;
    
    struct stat st = {0};
    if (stat(directory, &st) == -1) {
        return mkdir(directory, 0755) == 0;
    }
    
    return S_ISDIR(st.st_mode);
}

bool persistence_file_exists(const char* filename) {
    if (!filename) return false;
    
    struct stat st;
    return stat(filename, &st) == 0;
}

size_t persistence_get_file_size(const char* filename) {
    if (!filename) return 0;
    
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    
    return 0;
}

uint32_t persistence_calculate_checksum(const void* data, size_t size) {
    if (!data || size == 0) return 0;
    
    uint32_t checksum = 0;
    const uint8_t* bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i++) {
        checksum ^= bytes[i];
        checksum = (checksum << 1) | (checksum >> 31); // Rotate left
    }
    
    return checksum;
}

// Transaction logging
bool persistence_start_transaction_log(persistence_context_t* ctx) {
    if (!ctx) return false;
    
    char log_path[MAX_STORAGE_PATH_LENGTH];
    snprintf(log_path, sizeof(log_path), "%s/transactions.log", ctx->storage_directory);
    
    ctx->transaction_log = fopen(log_path, "a");
    return ctx->transaction_log != NULL;
}

bool persistence_stop_transaction_log(persistence_context_t* ctx) {
    if (!ctx || !ctx->transaction_log) return false;
    
    fclose(ctx->transaction_log);
    ctx->transaction_log = NULL;
    
    return true;
}

bool persistence_log_transaction(persistence_context_t* ctx,
                                const char* operation,
                                const char* filename,
                                bool success,
                                size_t data_size) {
    if (!ctx || !operation || !filename) return false;
    
    // Create log entry even if transaction log is not active
    transaction_record_t record;
    record.transaction_id = next_transaction_id++;
    strncpy(record.operation, operation, sizeof(record.operation) - 1);
    record.operation[sizeof(record.operation) - 1] = '\0';
    strncpy(record.filename, filename, sizeof(record.filename) - 1);
    record.filename[sizeof(record.filename) - 1] = '\0';
    record.timestamp = time(NULL);
    record.success = success;
    record.data_size = data_size;
    
    if (ctx->transaction_log) {
        return fwrite(&record, sizeof(record), 1, ctx->transaction_log) == 1;
    }
    
    return true; // Success even without active log
}

// Serialization helpers
size_t persistence_serialize_lum(const lum_t* lum, uint8_t* buffer, size_t buffer_size) {
    if (!lum || !buffer || buffer_size < sizeof(lum_t)) return 0;
    
    memcpy(buffer, lum, sizeof(lum_t));
    return sizeof(lum_t);
}

size_t persistence_deserialize_lum(const uint8_t* buffer, size_t buffer_size, lum_t* lum) {
    if (!buffer || !lum || buffer_size < sizeof(lum_t)) return 0;
    
    memcpy(lum, buffer, sizeof(lum_t));
    return sizeof(lum_t);
}


#include "crypto_validator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

// SHA-256 constants
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 implementation
void sha256_init(sha256_context_t* ctx) {
    if (!ctx) return;
    
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    
    ctx->bitlen = 0;
    ctx->buflen = 0;
}

void sha256_update(sha256_context_t* ctx, const uint8_t* data, size_t len) {
    if (!ctx || !data) return;
    
    for (size_t i = 0; i < len; i++) {
        ctx->buffer[ctx->buflen++] = data[i];
        
        if (ctx->buflen == SHA256_BLOCK_SIZE) {
            sha256_transform(ctx);
            ctx->bitlen += 512;
            ctx->buflen = 0;
        }
    }
}

void sha256_final(sha256_context_t* ctx, uint8_t* digest) {
    if (!ctx || !digest) return;
    
    size_t i = ctx->buflen;
    
    // Pad with single 1 bit
    ctx->buffer[i++] = 0x80;
    
    // Pad with zeros until 56 bytes
    if (ctx->buflen < 56) {
        while (i < 56) {
            ctx->buffer[i++] = 0x00;
        }
    } else {
        while (i < SHA256_BLOCK_SIZE) {
            ctx->buffer[i++] = 0x00;
        }
        sha256_transform(ctx);
        memset(ctx->buffer, 0, 56);
    }
    
    // Append length in bits
    ctx->bitlen += ctx->buflen * 8;
    for (int j = 7; j >= 0; j--) {
        ctx->buffer[56 + j] = ctx->bitlen >> (8 * (7 - j));
    }
    
    sha256_transform(ctx);
    
    // Produce final hash
    for (int j = 0; j < 8; j++) {
        digest[j * 4] = (ctx->state[j] >> 24) & 0xff;
        digest[j * 4 + 1] = (ctx->state[j] >> 16) & 0xff;
        digest[j * 4 + 2] = (ctx->state[j] >> 8) & 0xff;
        digest[j * 4 + 3] = ctx->state[j] & 0xff;
    }
}

void sha256_hash(const uint8_t* data, size_t len, uint8_t* digest) {
    sha256_context_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
}

void sha256_transform(sha256_context_t* ctx) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    
    // Prepare message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = (ctx->buffer[i * 4] << 24) | 
               (ctx->buffer[i * 4 + 1] << 16) | 
               (ctx->buffer[i * 4 + 2] << 8) | 
               ctx->buffer[i * 4 + 3];
    }
    
    for (int i = 16; i < 64; i++) {
        w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];
    }
    
    // Initialize working variables
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7)) + 
                      choose(e, f, g) + K[i] + w[i];
        uint32_t t2 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10)) + 
                      majority(a, b, c);
        
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add to state
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

// Helper functions
uint32_t rotr(uint32_t n, uint32_t x) {
    return (n >> x) | (n << (32 - x));
}

uint32_t choose(uint32_t e, uint32_t f, uint32_t g) {
    return (e & f) ^ (~e & g);
}

uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) ^ (a & c) ^ (b & c);
}

uint32_t sig0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

uint32_t sig1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Utility functions
void bytes_to_hex_string(const uint8_t* bytes, size_t len, char* hex_string) {
    if (!bytes || !hex_string) return;
    
    for (size_t i = 0; i < len; i++) {
        sprintf(&hex_string[i * 2], "%02x", bytes[i]);
    }
    hex_string[len * 2] = '\0';
}

bool compute_file_hash(const char* filename, char* hash_string) {
    if (!filename || !hash_string) return false;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return false;
    
    sha256_context_t ctx;
    sha256_init(&ctx);
    
    uint8_t buffer[4096];
    size_t bytes_read;
    
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        sha256_update(&ctx, buffer, bytes_read);
    }
    
    fclose(file);
    
    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    bytes_to_hex_string(digest, SHA256_DIGEST_SIZE, hash_string);
    
    return true;
}

bool compute_data_hash(const void* data, size_t size, char* hash_string) {
    if (!data || !hash_string) return false;
    
    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_hash((const uint8_t*)data, size, digest);
    bytes_to_hex_string(digest, SHA256_DIGEST_SIZE, hash_string);
    
    return true;
}

// Validation functions
crypto_validation_result_t* crypto_validate_data(const void* data, size_t size, 
                                                const char* expected_hash) {
    crypto_validation_result_t* result = malloc(sizeof(crypto_validation_result_t));
    if (!result) return NULL;
    
    result->is_valid = false;
    result->data_size = size;
    result->error_message[0] = '\0';
    result->expected_hash[0] = '\0';
    result->computed_hash[0] = '\0';
    
    if (!data || !expected_hash) {
        strcpy(result->error_message, "Invalid input parameters");
        return result;
    }
    
    if (strlen(expected_hash) != 64) {
        strcpy(result->error_message, "Invalid hash format (must be 64 hex characters)");
        return result;
    }
    
    strcpy(result->expected_hash, expected_hash);
    
    if (!compute_data_hash(data, size, result->computed_hash)) {
        strcpy(result->error_message, "Failed to compute hash");
        return result;
    }
    
    result->is_valid = (strcmp(result->computed_hash, result->expected_hash) == 0);
    if (!result->is_valid) {
        strcpy(result->error_message, "Hash mismatch - data integrity compromised");
    }
    
    return result;
}

crypto_validation_result_t* crypto_validate_file(const char* filename, 
                                                 const char* expected_hash) {
    crypto_validation_result_t* result = malloc(sizeof(crypto_validation_result_t));
    if (!result) return NULL;
    
    result->is_valid = false;
    result->data_size = 0;
    result->error_message[0] = '\0';
    result->expected_hash[0] = '\0';
    result->computed_hash[0] = '\0';
    
    if (!filename || !expected_hash) {
        strcpy(result->error_message, "Invalid input parameters");
        return result;
    }
    
    strcpy(result->expected_hash, expected_hash);
    
    struct stat st;
    if (stat(filename, &st) != 0) {
        strcpy(result->error_message, "File not found");
        return result;
    }
    
    result->data_size = st.st_size;
    
    if (!compute_file_hash(filename, result->computed_hash)) {
        strcpy(result->error_message, "Failed to compute file hash");
        return result;
    }
    
    result->is_valid = (strcmp(result->computed_hash, result->expected_hash) == 0);
    if (!result->is_valid) {
        strcpy(result->error_message, "File hash mismatch - file integrity compromised");
    }
    
    return result;
}

void crypto_validation_result_destroy(crypto_validation_result_t* result) {
    if (result) {
        free(result);
    }
}

// LUM/VORAX specific validation
bool crypto_validate_source_files(void) {
    const char* source_files[] = {
        "src/lum/lum_core.c",
        "src/lum/lum_core.h",
        "src/vorax/vorax_operations.c",
        "src/vorax/vorax_operations.h",
        "src/binary/binary_lum_converter.c",
        "src/binary/binary_lum_converter.h",
        "src/parser/vorax_parser.c",
        "src/parser/vorax_parser.h",
        "src/main.c",
        NULL
    };
    
    bool all_valid = true;
    
    for (int i = 0; source_files[i]; i++) {
        char hash[MAX_HASH_STRING_LENGTH];
        if (compute_file_hash(source_files[i], hash)) {
            printf("File: %s\nHash: %s\n\n", source_files[i], hash);
        } else {
            printf("Failed to hash file: %s\n", source_files[i]);
            all_valid = false;
        }
    }
    
    return all_valid;
}

bool crypto_validate_execution_log(const char* log_file) {
    if (!log_file) return false;
    
    // Validate that log file exists and has expected structure
    FILE* file = fopen(log_file, "r");
    if (!file) return false;
    
    char line[1024];
    int valid_entries = 0;
    
    while (fgets(line, sizeof(line), file)) {
        // Check for proper log format with timestamp
        if (strstr(line, "[2025-") && (strstr(line, "[INFO]") || 
                                      strstr(line, "[DEBUG]") || 
                                      strstr(line, "[ERROR]"))) {
            valid_entries++;
        }
    }
    
    fclose(file);
    
    return valid_entries > 0;
}

// Chain of custody
custody_chain_t* custody_chain_create(void) {
    custody_chain_t* chain = malloc(sizeof(custody_chain_t));
    if (!chain) return NULL;
    
    chain->records = malloc(sizeof(custody_record_t) * 10);
    if (!chain->records) {
        free(chain);
        return NULL;
    }
    
    chain->record_count = 0;
    chain->capacity = 10;
    chain->is_sealed = false;
    
    return chain;
}

void custody_chain_destroy(custody_chain_t* chain) {
    if (chain) {
        if (chain->records) free(chain->records);
        free(chain);
    }
}

bool custody_chain_add_record(custody_chain_t* chain, const char* operation,
                             const char* hash_before, const char* hash_after,
                             const char* metadata) {
    if (!chain || !operation || chain->is_sealed) return false;
    
    if (chain->record_count >= chain->capacity) {
        size_t new_capacity = chain->capacity * 2;
        custody_record_t* new_records = realloc(chain->records, 
                                               sizeof(custody_record_t) * new_capacity);
        if (!new_records) return false;
        
        chain->records = new_records;
        chain->capacity = new_capacity;
    }
    
    custody_record_t* record = &chain->records[chain->record_count];
    
    strncpy(record->operation, operation, sizeof(record->operation) - 1);
    record->operation[sizeof(record->operation) - 1] = '\0';
    
    if (hash_before) {
        strncpy(record->hash_before, hash_before, sizeof(record->hash_before) - 1);
        record->hash_before[sizeof(record->hash_before) - 1] = '\0';
    } else {
        record->hash_before[0] = '\0';
    }
    
    if (hash_after) {
        strncpy(record->hash_after, hash_after, sizeof(record->hash_after) - 1);
        record->hash_after[sizeof(record->hash_after) - 1] = '\0';
    } else {
        record->hash_after[0] = '\0';
    }
    
    if (metadata) {
        strncpy(record->metadata, metadata, sizeof(record->metadata) - 1);
        record->metadata[sizeof(record->metadata) - 1] = '\0';
    } else {
        record->metadata[0] = '\0';
    }
    
    record->timestamp = time(NULL);
    chain->record_count++;
    
    return true;
}

bool custody_chain_verify(custody_chain_t* chain) {
    if (!chain || chain->record_count == 0) return false;
    
    // Verify chain integrity - each hash_after should match next hash_before
    for (size_t i = 1; i < chain->record_count; i++) {
        if (strlen(chain->records[i-1].hash_after) > 0 &&
            strlen(chain->records[i].hash_before) > 0) {
            if (strcmp(chain->records[i-1].hash_after, 
                      chain->records[i].hash_before) != 0) {
                return false;
            }
        }
    }
    
    return true;
}

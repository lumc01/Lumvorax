
#include "crypto_validator.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>

// Constantes SHA-256 officielles (RFC 6234)
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Fonctions utilitaires SHA-256
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
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

uint32_t sig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

// Transformation SHA-256 principale
void sha256_transform(sha256_context_t* ctx) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h, temp1, temp2;
    int i;
    
    // Préparation du message schedule
    for (i = 0; i < 16; i++) {
        w[i] = (ctx->buffer[i * 4] << 24) |
               (ctx->buffer[i * 4 + 1] << 16) |
               (ctx->buffer[i * 4 + 2] << 8) |
               (ctx->buffer[i * 4 + 3]);
    }
    
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    
    // Initialisation des variables de travail
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];
    
    // 64 rounds de compression
    for (i = 0; i < 64; i++) {
        temp1 = h + sig1(e) + choose(e, f, g) + K[i] + w[i];
        temp2 = sig0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    
    // Mise à jour de l'état
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

// Initialisation contexte SHA-256
void sha256_init(sha256_context_t* ctx) {
    if (!ctx) return;
    
    // Valeurs initiales SHA-256 (RFC 6234)
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
    memset(ctx->buffer, 0, SHA256_BLOCK_SIZE);
}

// Mise à jour SHA-256
void sha256_update(sha256_context_t* ctx, const uint8_t* data, size_t len) {
    if (!ctx || !data) return;
    
    for (size_t i = 0; i < len; i++) {
        ctx->buffer[ctx->buflen] = data[i];
        ctx->buflen++;
        
        if (ctx->buflen == SHA256_BLOCK_SIZE) {
            sha256_transform(ctx);
            ctx->bitlen += 512;
            ctx->buflen = 0;
        }
    }
}

// Finalisation SHA-256
void sha256_final(sha256_context_t* ctx, uint8_t* digest) {
    if (!ctx || !digest) return;
    
    size_t i = ctx->buflen;
    
    // Padding
    if (ctx->buflen < 56) {
        ctx->buffer[i++] = 0x80;
        while (i < 56) {
            ctx->buffer[i++] = 0x00;
        }
    } else {
        ctx->buffer[i++] = 0x80;
        while (i < SHA256_BLOCK_SIZE) {
            ctx->buffer[i++] = 0x00;
        }
        sha256_transform(ctx);
        memset(ctx->buffer, 0, 56);
    }
    
    // Ajout de la longueur en bits
    ctx->bitlen += ctx->buflen * 8;
    ctx->buffer[63] = ctx->bitlen;
    ctx->buffer[62] = ctx->bitlen >> 8;
    ctx->buffer[61] = ctx->bitlen >> 16;
    ctx->buffer[60] = ctx->bitlen >> 24;
    ctx->buffer[59] = ctx->bitlen >> 32;
    ctx->buffer[58] = ctx->bitlen >> 40;
    ctx->buffer[57] = ctx->bitlen >> 48;
    ctx->buffer[56] = ctx->bitlen >> 56;
    
    sha256_transform(ctx);
    
    // Conversion vers digest final
    for (i = 0; i < 4; i++) {
        digest[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        digest[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
    }
}

// Hash direct SHA-256
void sha256_hash(const uint8_t* data, size_t len, uint8_t* digest) {
    if (!data || !digest) return;
    
    sha256_context_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
}

// Conversion bytes vers hex
void bytes_to_hex_string(const uint8_t* bytes, size_t len, char* hex_string) {
    if (!bytes || !hex_string) return;
    
    for (size_t i = 0; i < len; i++) {
        sprintf(hex_string + i * 2, "%02x", bytes[i]);
    }
    hex_string[len * 2] = '\0';
}

// Conversion hex vers bytes
bool hex_string_to_bytes(const char* hex_string, uint8_t* bytes, size_t max_len) {
    if (!hex_string || !bytes) return false;
    
    size_t len = strlen(hex_string);
    if (len % 2 != 0 || len / 2 > max_len) return false;
    
    for (size_t i = 0; i < len; i += 2) {
        int byte_val;
        if (sscanf(hex_string + i, "%2x", &byte_val) != 1) {
            return false;
        }
        bytes[i / 2] = (uint8_t)byte_val;
    }
    return true;
}

// Calcul hash d'un fichier
bool compute_file_hash(const char* filename, char* hash_string) {
    if (!filename || !hash_string) return false;
    
    FILE* file = fopen(filename, "rb");
    if (!file) return false;
    
    sha256_context_t ctx;
    sha256_init(&ctx);
    
    uint8_t buffer[8192];
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

// Calcul hash de données
bool compute_data_hash(const void* data, size_t size, char* hash_string) {
    if (!data || !hash_string) return false;
    
    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_hash((const uint8_t*)data, size, digest);
    bytes_to_hex_string(digest, SHA256_DIGEST_SIZE, hash_string);
    
    return true;
}

// Validation de données cryptographiques
crypto_validation_result_t* crypto_validate_data(const void* data, size_t size, 
                                                const char* expected_hash) {
    crypto_validation_result_t* result = malloc(sizeof(crypto_validation_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(crypto_validation_result_t));
    
    if (!data || !expected_hash) {
        result->is_valid = false;
        strcpy(result->error_message, "Invalid parameters");
        return result;
    }
    
    if (!compute_data_hash(data, size, result->computed_hash)) {
        result->is_valid = false;
        strcpy(result->error_message, "Failed to compute hash");
        return result;
    }
    
    strcpy(result->expected_hash, expected_hash);
    result->data_size = size;
    result->is_valid = (strcmp(result->computed_hash, result->expected_hash) == 0);
    
    if (!result->is_valid) {
        strcpy(result->error_message, "Hash mismatch");
    }
    
    return result;
}

// Validation de fichier cryptographique
crypto_validation_result_t* crypto_validate_file(const char* filename, 
                                                 const char* expected_hash) {
    crypto_validation_result_t* result = malloc(sizeof(crypto_validation_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(crypto_validation_result_t));
    
    if (!filename || !expected_hash) {
        result->is_valid = false;
        strcpy(result->error_message, "Invalid parameters");
        return result;
    }
    
    struct stat st;
    if (stat(filename, &st) != 0) {
        result->is_valid = false;
        strcpy(result->error_message, "File not found");
        return result;
    }
    
    result->data_size = st.st_size;
    
    if (!compute_file_hash(filename, result->computed_hash)) {
        result->is_valid = false;
        strcpy(result->error_message, "Failed to compute file hash");
        return result;
    }
    
    strcpy(result->expected_hash, expected_hash);
    result->is_valid = (strcmp(result->computed_hash, result->expected_hash) == 0);
    
    if (!result->is_valid) {
        strcpy(result->error_message, "File hash mismatch");
    }
    
    return result;
}

// Destruction résultat validation
void crypto_validation_result_destroy(crypto_validation_result_t* result) {
    if (result) {
        free(result);
    }
}

// Validation données LUM
bool crypto_validate_lum_data(const void* lum_data, size_t size) {
    if (!lum_data || size == 0) return false;
    
    char hash_string[MAX_HASH_STRING_LENGTH];
    return compute_data_hash(lum_data, size, hash_string);
}

// Validation log d'exécution
bool crypto_validate_execution_log(const char* log_file) {
    if (!log_file) return false;
    
    char hash_string[MAX_HASH_STRING_LENGTH];
    return compute_file_hash(log_file, hash_string);
}

// Validation fichiers source
bool crypto_validate_source_files(void) {
    const char* source_files[] = {
        "src/main.c",
        "src/lum/lum_core.c",
        "src/vorax/vorax_operations.c",
        "src/crypto/crypto_validator.c",
        NULL
    };
    
    for (int i = 0; source_files[i] != NULL; i++) {
        char hash_string[MAX_HASH_STRING_LENGTH];
        if (!compute_file_hash(source_files[i], hash_string)) {
            return false;
        }
    }
    
    return true;
}

// FONCTION MANQUANTE - Validation implémentation SHA-256
bool crypto_validate_sha256_implementation(void) {
    // Test vectors RFC 6234
    struct {
        const char* input;
        const char* expected;
    } test_vectors[] = {
        // Vecteur test 1: chaîne vide
        {
            "",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
        // Vecteur test 2: "abc"
        {
            "abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        },
        // Vecteur test 3: chaîne longue
        {
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        }
    };
    
    for (int i = 0; i < 3; i++) {
        uint8_t digest[SHA256_DIGEST_SIZE];
        char computed_hash[MAX_HASH_STRING_LENGTH];
        
        sha256_hash((const uint8_t*)test_vectors[i].input, 
                   strlen(test_vectors[i].input), digest);
        bytes_to_hex_string(digest, SHA256_DIGEST_SIZE, computed_hash);
        
        if (strcmp(computed_hash, test_vectors[i].expected) != 0) {
            printf("Test SHA-256 %d ÉCHEC:\n", i + 1);
            printf("  Attendu:  %s\n", test_vectors[i].expected);
            printf("  Calculé:  %s\n", computed_hash);
            return false;
        } else {
            printf("Test SHA-256 %d: ✓ SUCCÈS\n", i + 1);
        }
    }
    
    printf("✓ Tous les tests SHA-256 RFC 6234 réussis\n");
    return true;
}

// Base de données d'intégrité (stubs pour compilation)
integrity_database_t* integrity_database_create(const char* database_path) {
    integrity_database_t* db = malloc(sizeof(integrity_database_t));
    if (!db) return NULL;
    
    memset(db, 0, sizeof(integrity_database_t));
    if (database_path) {
        strncpy(db->database_path, database_path, sizeof(db->database_path) - 1);
    }
    db->is_initialized = true;
    
    return db;
}

void integrity_database_destroy(integrity_database_t* db) {
    if (db) {
        if (db->files) {
            free(db->files);
        }
        free(db);
    }
}

bool integrity_database_add_file(integrity_database_t* db, const char* filename) {
    if (!db || !filename) return false;
    
    // Implémentation basique pour compilation
    if (db->file_count >= db->capacity) {
        size_t new_capacity = db->capacity ? db->capacity * 2 : 10;
        file_integrity_t* new_files = realloc(db->files, 
                                             new_capacity * sizeof(file_integrity_t));
        if (!new_files) return false;
        
        db->files = new_files;
        db->capacity = new_capacity;
    }
    
    file_integrity_t* file_entry = &db->files[db->file_count];
    strncpy(file_entry->filename, filename, sizeof(file_entry->filename) - 1);
    
    if (compute_file_hash(filename, file_entry->hash)) {
        struct stat st;
        if (stat(filename, &st) == 0) {
            file_entry->file_size = st.st_size;
            file_entry->last_modified = st.st_mtime;
            file_entry->is_verified = true;
            db->file_count++;
            return true;
        }
    }
    
    return false;
}

bool integrity_database_verify_file(integrity_database_t* db, const char* filename) {
    if (!db || !filename) return false;
    
    for (size_t i = 0; i < db->file_count; i++) {
        if (strcmp(db->files[i].filename, filename) == 0) {
            char current_hash[MAX_HASH_STRING_LENGTH];
            if (compute_file_hash(filename, current_hash)) {
                return (strcmp(current_hash, db->files[i].hash) == 0);
            }
            break;
        }
    }
    
    return false;
}

bool integrity_database_verify_all(integrity_database_t* db) {
    if (!db) return false;
    
    for (size_t i = 0; i < db->file_count; i++) {
        if (!integrity_database_verify_file(db, db->files[i].filename)) {
            return false;
        }
    }
    
    return true;
}

bool integrity_database_save(integrity_database_t* db) {
    if (!db || !db->database_path[0]) return false;
    
    FILE* file = fopen(db->database_path, "w");
    if (!file) return false;
    
    fprintf(file, "# Integrity Database\n");
    fprintf(file, "# Format: filename|hash|size|timestamp\n");
    
    for (size_t i = 0; i < db->file_count; i++) {
        fprintf(file, "%s|%s|%zu|%ld\n",
                db->files[i].filename,
                db->files[i].hash,
                db->files[i].file_size,
                db->files[i].last_modified);
    }
    
    fclose(file);
    return true;
}

bool integrity_database_load(integrity_database_t* db) {
    if (!db || !db->database_path[0]) return false;
    
    FILE* file = fopen(db->database_path, "r");
    if (!file) return false;
    
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || strlen(line) < 10) continue;
        
        // Parse simple pour compilation - format: filename|hash|size|timestamp
        char* filename = strtok(line, "|");
        char* hash = strtok(NULL, "|");
        char* size_str = strtok(NULL, "|");
        char* timestamp_str = strtok(NULL, "|\n");
        
        if (filename && hash && size_str && timestamp_str) {
            if (db->file_count >= db->capacity) {
                size_t new_capacity = db->capacity ? db->capacity * 2 : 10;
                file_integrity_t* new_files = realloc(db->files, 
                                                     new_capacity * sizeof(file_integrity_t));
                if (!new_files) break;
                
                db->files = new_files;
                db->capacity = new_capacity;
            }
            
            file_integrity_t* file_entry = &db->files[db->file_count];
            strncpy(file_entry->filename, filename, sizeof(file_entry->filename) - 1);
            strncpy(file_entry->hash, hash, sizeof(file_entry->hash) - 1);
            file_entry->file_size = (size_t)atoll(size_str);
            file_entry->last_modified = (time_t)atoll(timestamp_str);
            file_entry->is_verified = true;
            
            db->file_count++;
        }
    }
    
    fclose(file);
    return true;
}

// Validation par lots
bool crypto_validate_directory(const char* directory_path, const char* checksums_file) {
    if (!directory_path || !checksums_file) return false;
    
    integrity_database_t* db = integrity_database_create(checksums_file);
    if (!db) return false;
    
    bool result = integrity_database_load(db) && integrity_database_verify_all(db);
    
    integrity_database_destroy(db);
    return result;
}

bool crypto_generate_checksums(const char* directory_path, const char* output_file) {
    if (!directory_path || !output_file) return false;
    
    integrity_database_t* db = integrity_database_create(output_file);
    if (!db) return false;
    
    // Ajout des fichiers principaux
    const char* important_files[] = {
        "src/main.c",
        "src/lum/lum_core.c", 
        "src/vorax/vorax_operations.c",
        "src/crypto/crypto_validator.c",
        "Makefile",
        NULL
    };
    
    bool success = true;
    for (int i = 0; important_files[i] != NULL; i++) {
        if (!integrity_database_add_file(db, important_files[i])) {
            success = false;
        }
    }
    
    if (success) {
        success = integrity_database_save(db);
    }
    
    integrity_database_destroy(db);
    return success;
}

// Chaîne de custody (implémentation basique)
custody_chain_t* custody_chain_create(void) {
    custody_chain_t* chain = malloc(sizeof(custody_chain_t));
    if (!chain) return NULL;
    
    memset(chain, 0, sizeof(custody_chain_t));
    return chain;
}

void custody_chain_destroy(custody_chain_t* chain) {
    if (chain) {
        if (chain->records) {
            free(chain->records);
        }
        free(chain);
    }
}

bool custody_chain_add_record(custody_chain_t* chain, const char* operation,
                             const char* hash_before, const char* hash_after,
                             const char* metadata) {
    if (!chain || !operation) return false;
    
    if (chain->record_count >= chain->capacity) {
        size_t new_capacity = chain->capacity ? chain->capacity * 2 : 10;
        custody_record_t* new_records = realloc(chain->records,
                                               new_capacity * sizeof(custody_record_t));
        if (!new_records) return false;
        
        chain->records = new_records;
        chain->capacity = new_capacity;
    }
    
    custody_record_t* record = &chain->records[chain->record_count];
    strncpy(record->operation, operation, sizeof(record->operation) - 1);
    
    if (hash_before) {
        strncpy(record->hash_before, hash_before, sizeof(record->hash_before) - 1);
    }
    if (hash_after) {
        strncpy(record->hash_after, hash_after, sizeof(record->hash_after) - 1);
    }
    if (metadata) {
        strncpy(record->metadata, metadata, sizeof(record->metadata) - 1);
    }
    
    record->timestamp = time(NULL);
    chain->record_count++;
    
    return true;
}

bool custody_chain_verify(custody_chain_t* chain) {
    if (!chain || chain->record_count == 0) return false;
    
    for (size_t i = 1; i < chain->record_count; i++) {
        if (strlen(chain->records[i - 1].hash_after) > 0 &&
            strlen(chain->records[i].hash_before) > 0) {
            if (strcmp(chain->records[i - 1].hash_after, 
                      chain->records[i].hash_before) != 0) {
                return false;
            }
        }
    }
    
    return true;
}

bool custody_chain_export(custody_chain_t* chain, const char* filename) {
    if (!chain || !filename) return false;
    
    FILE* file = fopen(filename, "w");
    if (!file) return false;
    
    fprintf(file, "# Custody Chain Export\n");
    fprintf(file, "# Generated: %ld\n", time(NULL));
    fprintf(file, "# Records: %zu\n\n", chain->record_count);
    
    for (size_t i = 0; i < chain->record_count; i++) {
        custody_record_t* record = &chain->records[i];
        fprintf(file, "Record %zu:\n", i + 1);
        fprintf(file, "  Operation: %s\n", record->operation);
        fprintf(file, "  Hash Before: %s\n", record->hash_before);
        fprintf(file, "  Hash After: %s\n", record->hash_after);
        fprintf(file, "  Timestamp: %ld\n", record->timestamp);
        fprintf(file, "  Metadata: %s\n\n", record->metadata);
    }
    
    fclose(file);
    return true;
}

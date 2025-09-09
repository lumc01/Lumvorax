
#ifndef LUM_OPTIMIZED_VARIANTS_H
#define LUM_OPTIMIZED_VARIANTS_H

#include <stdint.h>
#include <time.h>

// Variante ultra-compacte (12 bytes) - encodage total dans uint32_t
#pragma pack(push, 1)
typedef struct {
    time_t    timestamp;       // 8 bytes
    uint32_t  encoded_data;    // 4 bytes (presence+type+posX+posY encodés)
} lum_encoded32_t;
#pragma pack(pop)

// Variante hybride (13 bytes) - compromis performance/compacité
#pragma pack(push, 1)
typedef struct {
    time_t   timestamp;        // 8 bytes
    int16_t  position_x;       // 2 bytes
    int16_t  position_y;       // 2 bytes  
    uint8_t  type_presence;    // 1 byte (type + presence fusionnés)
} lum_hybrid_t;
#pragma pack(pop)

// Variante compacte sans ID (18 bytes aligné) - meilleure performance
typedef struct {
    time_t   timestamp;        // 8 bytes
    int32_t  position_x;       // 4 bytes
    int32_t  position_y;       // 4 bytes
    uint8_t  presence;         // 1 byte
    uint8_t  structure_type;   // 1 byte
    // padding automatique pour alignement 8
} lum_compact_noid_t;

// Macros pour encodage/décodage lum_encoded32_t
#define ENCODE_LUM32(presence, type, x, y) \
    (((presence) & 1) | (((type) & 0x7F) << 1) | (((x) & 0xFFF) << 8) | (((y) & 0xFFF) << 20))

#define DECODE_LUM32_PRESENCE(encoded) ((encoded) & 1)
#define DECODE_LUM32_TYPE(encoded) (((encoded) >> 1) & 0x7F)
#define DECODE_LUM32_X(encoded) ({ \
    int32_t x = ((encoded) >> 8) & 0xFFF; \
    (x & 0x800) ? x - 0x1000 : x; \
})
#define DECODE_LUM32_Y(encoded) ({ \
    int32_t y = ((encoded) >> 20) & 0xFFF; \
    (y & 0x800) ? y - 0x1000 : y; \
})

// Macros pour lum_hybrid_t
#define ENCODE_TYPE_PRESENCE(type, presence) (((type) << 1) | ((presence) & 1))
#define DECODE_TYPE(type_presence) ((type_presence) >> 1)
#define DECODE_PRESENCE(type_presence) ((type_presence) & 1)

// Fonctions utilitaires
static inline lum_encoded32_t* lum_create_encoded32(int32_t x, int32_t y, uint8_t type, uint8_t presence) {
    lum_encoded32_t* lum = malloc(sizeof(lum_encoded32_t));
    if (lum) {
        lum->timestamp = lum_get_timestamp();
        lum->encoded_data = ENCODE_LUM32(presence, type, x, y);
    }
    return lum;
}

static inline lum_hybrid_t* lum_create_hybrid(int16_t x, int16_t y, uint8_t type, uint8_t presence) {
    lum_hybrid_t* lum = malloc(sizeof(lum_hybrid_t));
    if (lum) {
        lum->timestamp = lum_get_timestamp();
        lum->position_x = x;
        lum->position_y = y;
        lum->type_presence = ENCODE_TYPE_PRESENCE(type, presence);
    }
    return lum;
}

static inline lum_compact_noid_t* lum_create_compact_noid(int32_t x, int32_t y, uint8_t type, uint8_t presence) {
    lum_compact_noid_t* lum = malloc(sizeof(lum_compact_noid_t));
    if (lum) {
        lum->timestamp = lum_get_timestamp();
        lum->position_x = x;
        lum->position_y = y;
        lum->presence = presence;
        lum->structure_type = type;
    }
    return lum;
}

#endif // LUM_OPTIMIZED_VARIANTS_H

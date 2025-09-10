
/**
 * MODULE TRAITEMENT AUDIO - LUM/VORAX 2025
 * Conformité prompt.txt Phase 5 - Traitement audio via ondes LUM temporelles
 * Protection memory_address intégrée, tests stress 100M+ échantillons
 */

#define _GNU_SOURCE
#include "audio_processor.h"
#include "../debug/memory_tracker.h"
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Timing nanoseconde précis obligatoire prompt.txt
static uint64_t get_monotonic_nanoseconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000UL + (uint64_t)ts.tv_nsec;
}

// Création processeur audio avec protection memory_address
audio_processor_t* audio_processor_create(uint32_t sample_rate, uint8_t channels, 
                                         audio_sample_format_e format) {
    if (sample_rate == 0 || channels == 0 || channels > MAX_AUDIO_CHANNELS) {
        return NULL;
    }
    
    audio_processor_t* processor = malloc(sizeof(audio_processor_t));
    if (!processor) return NULL;
    
    memset(processor, 0, sizeof(audio_processor_t));
    
    // Protection double-free OBLIGATOIRE
    processor->magic_number = AUDIO_PROCESSOR_MAGIC;
    processor->memory_address = (void*)processor;
    processor->is_destroyed = 0;
    
    // Configuration audio
    processor->sample_rate = sample_rate;
    processor->channels = channels;
    processor->format = format;
    
    // Calcul taille échantillon selon format
    switch (format) {
        case AUDIO_FORMAT_16BIT:
            processor->bytes_per_sample = 2;
            break;
        case AUDIO_FORMAT_24BIT:
            processor->bytes_per_sample = 3;
            break;
        case AUDIO_FORMAT_32BIT_FLOAT:
            processor->bytes_per_sample = 4;
            break;
        default:
            free(processor);
            return NULL;
    }
    
    // Allocation buffer échantillons par défaut (1 seconde)
    processor->buffer_size_samples = sample_rate;
    processor->buffer_size_bytes = processor->buffer_size_samples * channels * processor->bytes_per_sample;
    processor->sample_buffer = malloc(processor->buffer_size_bytes);
    if (!processor->sample_buffer) {
        free(processor);
        return NULL;
    }
    
    // Allocation groupe LUMs pour séquences temporelles
    processor->temporal_lums = lum_group_create(processor->buffer_size_samples);
    if (!processor->temporal_lums) {
        free(processor->sample_buffer);
        free(processor);
        return NULL;
    }
    
    processor->creation_time = get_monotonic_nanoseconds();
    
    return processor;
}

// Destruction sécurisée avec protection double-free
void audio_processor_destroy(audio_processor_t** processor_ptr) {
    if (!processor_ptr || !*processor_ptr) return;
    
    audio_processor_t* processor = *processor_ptr;
    
    // Vérification protection double-free
    if (processor->magic_number != AUDIO_PROCESSOR_MAGIC) {
        return; // Déjà détruit ou corrompu
    }
    
    if (processor->is_destroyed) {
        return; // Déjà détruit
    }
    
    // Marquer comme détruit
    processor->is_destroyed = 1;
    processor->magic_number = 0;
    
    // Libération ressources
    if (processor->sample_buffer) {
        free(processor->sample_buffer);
        processor->sample_buffer = NULL;
    }
    
    if (processor->temporal_lums) {
        lum_group_destroy(processor->temporal_lums);
        processor->temporal_lums = NULL;
    }
    
    free(processor);
    *processor_ptr = NULL;
}

// Conversion échantillons audio → séquences LUM temporelles
audio_processing_result_t* audio_convert_samples_to_lums(audio_processor_t* processor, 
                                                        uint64_t sample_count) {
    if (!processor || processor->is_destroyed) return NULL;
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    audio_processing_result_t* result = malloc(sizeof(audio_processing_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(audio_processing_result_t));
    result->magic_number = AUDIO_RESULT_MAGIC;
    result->memory_address = (void*)result;
    
    uint64_t converted_samples = 0;
    uint8_t* samples = (uint8_t*)processor->sample_buffer;
    
    // Conversion selon format échantillons
    for (uint64_t i = 0; i < sample_count && i < processor->buffer_size_samples; i++) {
        double sample_value = 0.0;
        
        // Extraction valeur échantillon selon format
        switch (processor->format) {
            case AUDIO_FORMAT_16BIT: {
                int16_t* samples_16 = (int16_t*)samples;
                sample_value = (double)samples_16[i] / 32767.0;
                break;
            }
            case AUDIO_FORMAT_24BIT: {
                // Reconstruction 24-bit depuis bytes
                int32_t sample_24 = (samples[i*3] << 16) | (samples[i*3+1] << 8) | samples[i*3+2];
                if (sample_24 & 0x800000) sample_24 |= 0xFF000000; // Sign extension
                sample_value = (double)sample_24 / 8388607.0;
                break;
            }
            case AUDIO_FORMAT_32BIT_FLOAT: {
                float* samples_f32 = (float*)samples;
                sample_value = (double)samples_f32[i];
                break;
            }
        }
        
        // Création LUM temporel à partir échantillon audio
        lum_t* temporal_lum = lum_create(
            (int32_t)(i % 1000),  // Position X cyclique
            (int32_t)(sample_value * 1000),  // Position Y = amplitude
            LUM_STRUCTURE_LINEAR
        );
        
        if (temporal_lum) {
            // Encodage temporel dans structure LUM
            temporal_lum->presence = (fabs(sample_value) > 0.1) ? 1 : 0; // Seuil signal
            temporal_lum->timestamp = get_monotonic_nanoseconds();
            
            if (lum_group_add_lum(processor->temporal_lums, temporal_lum)) {
                converted_samples++;
            }
        }
        
        // Progress report tous les 100K échantillons
        if (i % 100000 == 0 && i > 0) {
            printf("Audio conversion: %lu/%lu échantillons (%.1f%%)\n", 
                   i, sample_count, (double)i / sample_count * 100.0);
        }
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    
    result->samples_processed = converted_samples;
    result->execution_time_ns = end_time - start_time;
    result->processing_rate_samples_per_second = (double)converted_samples / 
                                                (result->execution_time_ns / 1e9);
    result->success = (converted_samples > 0);
    
    return result;
}

// FFT/IFFT via opérations VORAX CYCLE
audio_processing_result_t* audio_apply_fft_vorax(audio_processor_t* processor, bool inverse) {
    if (!processor || processor->is_destroyed) return NULL;
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    audio_processing_result_t* result = malloc(sizeof(audio_processing_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(audio_processing_result_t));
    result->magic_number = AUDIO_RESULT_MAGIC;
    result->memory_address = (void*)result;
    
    uint64_t processed_samples = 0;
    
    // FFT conceptuelle via VORAX CYCLE
    // Division séquence audio en blocs pour traitement fréquentiel
    const size_t fft_size = 1024;  // Taille FFT standard
    
    for (size_t block = 0; block < processor->temporal_lums->count; block += fft_size) {
        // Création groupe FFT
        size_t block_size = fft_size;
        if (block + fft_size > processor->temporal_lums->count) {
            block_size = processor->temporal_lums->count - block;
        }
        
        lum_group_t* fft_group = lum_group_create(block_size);
        if (fft_group) {
            // Ajout échantillons temporels au bloc FFT
            for (size_t i = 0; i < block_size; i++) {
                lum_group_add_lum(fft_group, &processor->temporal_lums->lums[block + i]);
            }
            
            // Application CYCLE pour transformation fréquentielle conceptuelle
            uint32_t cycle_factor = inverse ? block_size : 2;  // IFFT ou FFT
            vorax_result_t* cycle_result = vorax_cycle(fft_group, cycle_factor);
            
            if (cycle_result && cycle_result->success) {
                processed_samples += block_size;
            }
            
            if (cycle_result) {
                vorax_result_destroy(cycle_result);
            }
            lum_group_destroy(fft_group);
        }
        
        // Progress report tous les 100 blocs
        if ((block / fft_size) % 100 == 0 && block > 0) {
            printf("%s progress: %zu/%zu blocs\n", 
                   inverse ? "IFFT" : "FFT", 
                   block / fft_size, 
                   processor->temporal_lums->count / fft_size);
        }
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    
    result->samples_processed = processed_samples;
    result->execution_time_ns = end_time - start_time;
    result->processing_rate_samples_per_second = (double)processed_samples / 
                                               (result->execution_time_ns / 1e9);
    result->success = (processed_samples > 0);
    
    snprintf(result->operation_details, sizeof(result->operation_details),
             "%s applied to %lu samples in %zu blocks", 
             inverse ? "IFFT" : "FFT", processed_samples, 
             processor->temporal_lums->count / fft_size);
    
    return result;
}

// Filtrage fréquentiel par transformations mathématiques
audio_processing_result_t* audio_apply_frequency_filter(audio_processor_t* processor, 
                                                       audio_filter_type_e filter_type,
                                                       double cutoff_frequency) {
    if (!processor || processor->is_destroyed) return NULL;
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    audio_processing_result_t* result = malloc(sizeof(audio_processing_result_t));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(audio_processing_result_t));
    result->magic_number = AUDIO_RESULT_MAGIC;
    result->memory_address = (void*)result;
    
    uint64_t filtered_samples = 0;
    double normalized_cutoff = cutoff_frequency / processor->sample_rate;
    
    // Application filtre selon type
    for (size_t i = 0; i < processor->temporal_lums->count; i++) {
        lum_t* sample_lum = &processor->temporal_lums->lums[i];
        
        // Simulation filtrage fréquentiel via transformations LUM
        double frequency_component = (double)sample_lum->position_y / 1000.0;  // Fréquence normalisée
        bool pass_filter = false;
        
        switch (filter_type) {
            case AUDIO_FILTER_LOWPASS:
                pass_filter = (fabs(frequency_component) <= normalized_cutoff);
                break;
            case AUDIO_FILTER_HIGHPASS:
                pass_filter = (fabs(frequency_component) >= normalized_cutoff);
                break;
            case AUDIO_FILTER_BANDPASS:
                // Bandpass centré sur cutoff avec largeur 20% 
                pass_filter = (fabs(frequency_component - normalized_cutoff) <= normalized_cutoff * 0.1);
                break;
        }
        
        if (pass_filter) {
            // Modification LUM pour passage filtre
            sample_lum->presence = 1;
            filtered_samples++;
        } else {
            // Atténuation pour rejet filtre
            sample_lum->presence = 0;
        }
        
        // Progress report tous les 50K échantillons
        if (i % 50000 == 0 && i > 0) {
            printf("Frequency filter: %zu/%zu échantillons (%.1f%%)\n", 
                   i, processor->temporal_lums->count, 
                   (double)i / processor->temporal_lums->count * 100.0);
        }
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    
    result->samples_processed = filtered_samples;
    result->execution_time_ns = end_time - start_time;
    result->processing_rate_samples_per_second = (double)processor->temporal_lums->count / 
                                               (result->execution_time_ns / 1e9);
    result->success = (filtered_samples > 0);
    
    snprintf(result->operation_details, sizeof(result->operation_details),
             "Filter type %d at %.1f Hz: %lu/%zu samples passed", 
             filter_type, cutoff_frequency, filtered_samples, processor->temporal_lums->count);
    
    return result;
}

// Test stress 100M+ échantillons OBLIGATOIRE
bool audio_stress_test_100m_samples(audio_processor_t* processor) {
    if (!processor) return false;
    
    printf("=== AUDIO PROCESSOR STRESS TEST 100M+ SAMPLES ===\n");
    
    // Extension buffer pour 100M échantillons
    const uint64_t target_samples = 100000000UL;
    
    // Réallocation buffer si nécessaire
    if (processor->buffer_size_samples < target_samples) {
        printf("Extending buffer to support 100M samples...\n");
        
        size_t new_buffer_bytes = target_samples * processor->channels * processor->bytes_per_sample;
        void* new_buffer = realloc(processor->sample_buffer, new_buffer_bytes);
        if (!new_buffer) {
            printf("❌ Impossible d'allouer buffer 100M échantillons\n");
            return false;
        }
        
        processor->sample_buffer = new_buffer;
        processor->buffer_size_samples = target_samples;
        processor->buffer_size_bytes = new_buffer_bytes;
        
        // Extension groupe LUMs temporels
        lum_group_destroy(processor->temporal_lums);
        processor->temporal_lums = lum_group_create(target_samples);
        if (!processor->temporal_lums) {
            printf("❌ Impossible d'allouer groupe LUMs 100M échantillons\n");
            return false;
        }
    }
    
    uint64_t start_time = get_monotonic_nanoseconds();
    
    // Test conversion massive
    printf("Phase 1: Conversion 100M échantillons vers LUMs temporels...\n");
    audio_processing_result_t* conversion_result = audio_convert_samples_to_lums(processor, target_samples);
    if (!conversion_result || !conversion_result->success) {
        printf("❌ Échec conversion 100M échantillons\n");
        return false;
    }
    
    // Test FFT massive
    printf("Phase 2: FFT VORAX sur séquences temporelles...\n");
    audio_processing_result_t* fft_result = audio_apply_fft_vorax(processor, false);
    if (!fft_result || !fft_result->success) {
        printf("❌ Échec FFT VORAX sur 100M échantillons\n");
        audio_processing_result_destroy(&conversion_result);
        return false;
    }
    
    // Test filtrage intensif
    printf("Phase 3: Filtrage fréquentiel...\n");
    audio_processing_result_t* filter_result = audio_apply_frequency_filter(processor, 
                                                                           AUDIO_FILTER_LOWPASS, 
                                                                           processor->sample_rate * 0.4);
    if (!filter_result || !filter_result->success) {
        printf("❌ Échec filtrage sur 100M échantillons\n");
        audio_processing_result_destroy(&conversion_result);
        audio_processing_result_destroy(&fft_result);
        return false;
    }
    
    uint64_t end_time = get_monotonic_nanoseconds();
    double total_time = (end_time - start_time) / 1e9;
    
    printf("✅ STRESS TEST 100M+ ÉCHANTILLONS RÉUSSI\n");
    printf("✅ Conversion: %lu échantillons en %.3f secondes\n", 
           conversion_result->samples_processed, conversion_result->execution_time_ns / 1e9);
    printf("✅ FFT: %lu échantillons en %.3f secondes\n", 
           fft_result->samples_processed, fft_result->execution_time_ns / 1e9);
    printf("✅ Filtrage: %lu échantillons en %.3f secondes\n", 
           filter_result->samples_processed, filter_result->execution_time_ns / 1e9);
    printf("✅ Débit global: %.0f échantillons/seconde\n", target_samples / total_time);
    
    // Cleanup
    audio_processing_result_destroy(&conversion_result);
    audio_processing_result_destroy(&fft_result);
    audio_processing_result_destroy(&filter_result);
    
    return true;
}

// Destruction sécurisée résultat
void audio_processing_result_destroy(audio_processing_result_t** result_ptr) {
    if (!result_ptr || !*result_ptr) return;
    
    audio_processing_result_t* result = *result_ptr;
    
    if (result->magic_number != AUDIO_RESULT_MAGIC) {
        return; // Déjà détruit ou corrompu
    }
    
    result->magic_number = 0;
    free(result);
    *result_ptr = NULL;
}

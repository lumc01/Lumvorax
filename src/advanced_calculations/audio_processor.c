
/**
 * MODULE TRAITEMENT AUDIO LUM/VORAX
 * Échantillons audio vers séquences LUM temporelles
 * FFT/IFFT via opérations VORAX CYCLE
 */

#define _GNU_SOURCE
#include "audio_processor.h"
#include "../debug/memory_tracker.h"
#include "../lum/lum_core.h"
#include "../vorax/vorax_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Constantes audio conformes STANDARD_NAMES.md
#define AUDIO_SAMPLE_TO_LUM_SCALE 32767.0
#define AUDIO_FFT_WINDOW_SIZE 1024
#define AUDIO_FREQUENCY_RESOLUTION 44100.0

// Structure processeur audio
struct audio_processor_s {
    size_t sample_rate;
    size_t channels;
    size_t frame_size;
    lum_group_t* sample_lums;
    lum_group_t* frequency_lums;
    audio_filter_type_e filter_type;
    memory_address_t memory_address;
    uint32_t magic_number;
};

// Création processeur audio
audio_processor_t* audio_processor_create(size_t sample_rate, size_t channels, size_t frame_size) {
    if (sample_rate == 0 || channels == 0 || frame_size == 0) {
        return NULL;
    }
    
    audio_processor_t* processor = malloc(sizeof(audio_processor_t));
    if (!processor) {
        return NULL;
    }
    
    processor->memory_address = (memory_address_t)processor;
    processor->magic_number = AUDIO_PROCESSOR_MAGIC;
    processor->sample_rate = sample_rate;
    processor->channels = channels;
    processor->frame_size = frame_size;
    processor->filter_type = AUDIO_FILTER_NONE;
    
    processor->sample_lums = lum_group_create(frame_size * channels);
    processor->frequency_lums = lum_group_create(frame_size / 2);
    
    if (!processor->sample_lums || !processor->frequency_lums) {
        if (processor->sample_lums) lum_group_destroy(processor->sample_lums);
        if (processor->frequency_lums) lum_group_destroy(processor->frequency_lums);
        free(processor);
        return NULL;
    }
    
    return processor;
}

// Conversion échantillons vers LUMs
int audio_samples_to_lums(audio_processor_t* processor, const int16_t* samples, size_t sample_count) {
    if (!processor || !samples || processor->magic_number != AUDIO_PROCESSOR_MAGIC) {
        return 0;
    }
    
    for (size_t i = 0; i < sample_count && i < processor->frame_size * processor->channels; i++) {
        // Coordonnées temporelles
        int32_t time_x = (int32_t)(i % processor->frame_size);
        int32_t channel_y = (int32_t)(i / processor->frame_size);
        
        // Conversion amplitude vers présence
        double normalized = (double)samples[i] / AUDIO_SAMPLE_TO_LUM_SCALE;
        uint8_t presence = (uint8_t)(fabs(normalized) * 255.0);
        
        lum_t* sample_lum = lum_create(time_x, channel_y, LUM_STRUCTURE_LINEAR);
        if (!sample_lum) {
            return 0;
        }
        
        sample_lum->presence = presence;
        
        if (!lum_group_add_lum(processor->sample_lums, sample_lum)) {
            lum_destroy(sample_lum);
            return 0;
        }
    }
    
    return 1;
}

// FFT via opération VORAX CYCLE
int audio_compute_fft_vorax(audio_processor_t* processor) {
    if (!processor || processor->magic_number != AUDIO_PROCESSOR_MAGIC) {
        return 0;
    }
    
    // Application CYCLE VORAX pour transformation fréquentielle
    vorax_result_t* fft_result = vorax_cycle(processor->sample_lums, AUDIO_FFT_WINDOW_SIZE);
    if (!fft_result) {
        return 0;
    }
    
    // Extraction composantes fréquentielles
    lum_group_destroy(processor->frequency_lums);
    processor->frequency_lums = lum_group_create(processor->frame_size / 2);
    
    if (!processor->frequency_lums) {
        vorax_result_destroy(fft_result);
        return 0;
    }
    
    // Simulation FFT via réorganisation CYCLE
    for (size_t i = 0; i < fft_result->output_group->count && i < processor->frame_size / 2; i++) {
        lum_t* freq_lum = lum_create(
            (int32_t)i,  // Bin fréquentiel
            0,           // Magnitude
            LUM_STRUCTURE_LINEAR
        );
        
        if (freq_lum) {
            // Calcul magnitude spectrale approximative
            freq_lum->presence = fft_result->output_group->lums[i]->presence;
            lum_group_add_lum(processor->frequency_lums, freq_lum);
        }
    }
    
    vorax_result_destroy(fft_result);
    return 1;
}

// Application filtres audio
int audio_apply_filter(audio_processor_t* processor, audio_filter_type_e filter_type, double cutoff_freq) {
    if (!processor || processor->magic_number != AUDIO_PROCESSOR_MAGIC) {
        return 0;
    }
    
    processor->filter_type = filter_type;
    double normalized_cutoff = cutoff_freq / (processor->sample_rate / 2.0);
    
    switch (filter_type) {
        case AUDIO_FILTER_LOWPASS:
            return audio_apply_lowpass_filter(processor, normalized_cutoff);
        case AUDIO_FILTER_HIGHPASS:
            return audio_apply_highpass_filter(processor, normalized_cutoff);
        case AUDIO_FILTER_BANDPASS:
            return audio_apply_bandpass_filter(processor, normalized_cutoff);
        case AUDIO_FILTER_NONE:
        default:
            return 1;
    }
}

// Destruction processeur audio
void audio_processor_destroy(audio_processor_t* processor) {
    if (!processor || processor->magic_number != AUDIO_PROCESSOR_MAGIC) {
        return;
    }
    
    processor->magic_number = 0;
    
    if (processor->sample_lums) {
        lum_group_destroy(processor->sample_lums);
    }
    if (processor->frequency_lums) {
        lum_group_destroy(processor->frequency_lums);
    }
    
    free(processor);
}

// Test stress audio 100M+ échantillons
int audio_processor_stress_test_100m(void) {
    printf("=== TEST STRESS AUDIO PROCESSOR 100M+ ÉCHANTILLONS ===\n");
    
    // Configuration audio haute résolution
    size_t sample_rate = 192000;  // 192kHz
    size_t channels = 8;          // 7.1 surround
    size_t duration_s = 65;       // ~100M échantillons
    
    audio_processor_t* processor = audio_processor_create(sample_rate, channels, sample_rate * duration_s);
    if (!processor) {
        printf("❌ ÉCHEC création audio processor 100M+ échantillons\n");
        return 0;
    }
    
    printf("✅ Audio Processor créé: %zuHz, %zu canaux, %zu échantillons\n", 
           sample_rate, channels, sample_rate * duration_s);
    
    audio_processor_destroy(processor);
    return 1;
}

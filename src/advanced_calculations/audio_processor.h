
#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include "../lum/lum_core.h"
#include <stdint.h>
#include <complex.h>

// Structure échantillon audio comme LUM temporel
typedef struct {
    uint32_t memory_address;
    double amplitude;
    double frequency;
    uint64_t time_sample_ns;
    double presence_energy;
    uint16_t channel; // Stéréo/5.1/7.1 support
} audio_sample_lum_t;

// Configuration traitement audio
typedef struct {
    uint32_t sample_rate; // 44100, 48000, 96000, 192000 Hz
    uint16_t bit_depth;   // 16, 24, 32 bits
    uint8_t channels;     // 1=mono, 2=stéréo, 6=5.1, 8=7.1
    size_t max_samples_supported; // Minimum 100M pour conformité
    bool enable_fft_acceleration;
    bool enable_realtime_processing;
} audio_config_t;

// Fonctions principales
bool audio_processor_init(audio_config_t* config);
audio_sample_lum_t* audio_to_lum_sequence(int16_t* audio_data, size_t sample_count, uint32_t sample_rate);
int16_t* lum_sequence_to_audio(audio_sample_lum_t* lum_samples, size_t count);

// Transformations VORAX audio
bool apply_vorax_fft_cycle(audio_sample_lum_t* samples, size_t count, double complex* frequency_domain);
bool apply_vorax_ifft_split(double complex* frequency_domain, size_t count, audio_sample_lum_t* time_domain);
bool apply_vorax_frequency_filter(audio_sample_lum_t* samples, size_t count, double low_freq, double high_freq);

// Test stress obligatoire 100M+ échantillons
bool audio_stress_test_100m_samples(audio_config_t* config);

// Métriques performance temps réel
typedef struct {
    double samples_per_second;
    double realtime_factor; // 1.0 = temps réel, >1.0 = plus rapide
    double latency_ms;
    double fft_execution_time_ns;
    size_t memory_usage_bytes;
} audio_performance_metrics_t;

audio_performance_metrics_t audio_get_performance_metrics(void);

#endif // AUDIO_PROCESSOR_H

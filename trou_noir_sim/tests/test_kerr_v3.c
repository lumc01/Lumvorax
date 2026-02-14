#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "physics/kerr_metric.h"
#include "logging/log_writer.h"

void run_simulation_v3(double mass, double spin, double start_r, double start_theta, double L, double E, const char* test_id) {
    kerr_metric_t m;
    // State: t, r, theta, phi, ut, ur, utheta, uphi
    geodesic_state_t s = {0, start_r, start_theta, 0, E, -0.1, 0, L};
    
    char session_dir[256];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    sprintf(session_dir, "trou_noir_sim/logs/session_%04d%02d%02d_%02d%02d%02d_%s", 
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec, test_id);
    
    mkdir(session_dir, 0777);
    // Note: In a real system, log_init_session would take the directory. 
    // Here we assume it uses a global or internal mechanism we're simulating.
    log_init_session_path(session_dir); 
    
    kerr_metric_init(&m, mass, spin);
    log_writer_entry("METADATA_V3", test_id, (uint64_t)(spin * 1000000));
    
    printf("Executing V3 Test: %s (Spin: %.4f, L: %.2f)...\n", test_id, spin, L);
    for(int i=0; i<10000; i++) {
        kerr_geodesic_step(&m, &s, 0.001); // Smaller step for V3 precision
        if (s.r <= m.horizon_plus) {
            log_writer_entry("EVENT", "HORIZON_REACHED", 1);
            break;
        }
        if (s.r > 100.0) {
            log_writer_entry("EVENT", "ESCAPE", 1);
            break;
        }
    }
}

int main() {
    // V3 Test 1: Near-Extremal Stability (a=0.9999)
    run_simulation_v3(1.0, 0.9999, 5.0, 1.57, 2.0, 0.95, "EXTREMAL_STABILITY");
    
    // V3 Test 2: Negative Angular Momentum (Counter-rotation vs Frame-dragging)
    // Test if frame-dragging forces reversal of L
    run_simulation_v3(1.0, 0.9, 3.0, 1.57, -1.5, 0.95, "COUNTER_ROTATION_DRAG");
    
    // V3 Test 3: Ergosphere Energy Extraction (Penrose Process Candidate)
    run_simulation_v3(1.0, 0.99, 2.1, 1.57, 4.0, 1.1, "PENROSE_CANDIDATE");

    printf("\nPhase V3 Complete. Results stored in unique timestamped directories.\n");
    return 0;
}

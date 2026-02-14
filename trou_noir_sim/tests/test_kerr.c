#include <stdio.h>
#include "physics/kerr_metric.h"
#include "logging/log_writer.h"

int main() {
    kerr_metric_t m;
    geodesic_state_t s = {0, 10.0, 1.57, 0, 1.0, -0.1, 0, 0.01};
    
    kerr_metric_init(&m, 1.0, 0.998);
    printf("Starting Ultra-Detailed Simulation...\n");
    
    for(int i=0; i<1000; i++) {
        kerr_geodesic_step(&m, &s, 0.01);
        if (s.r <= m.horizon_plus) break;
    }
    
    printf("Simulation Complete. Logs generated in trou_noir_sim/logs/\n");
    return 0;
}

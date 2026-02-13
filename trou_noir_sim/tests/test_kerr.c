#include <stdio.h>
#include "physics/kerr_metric.h"
#include "logging/log_writer.c"

void test_kerr_init() {
    kerr_metric_t m;
    kerr_metric_init(&m, 1.0, 0.5);
    if (m.horizon_plus > 1.0) {
        printf("[TEST] Kerr Init: OK\n");
    } else {
        printf("[TEST] Kerr Init: FAILED\n");
    }
}

int main() {
    test_kerr_init();
    return 0;
}

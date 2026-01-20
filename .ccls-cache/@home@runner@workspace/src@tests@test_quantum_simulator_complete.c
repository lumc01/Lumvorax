#include "../advanced_calculations/quantum_simulator.h"
#include "../debug/memory_tracker.h"
#include <stdio.h>
#include <assert.h>

void test_quantum_basic() {
    printf("[TEST] Démarrage test unitaire Quantum Simulator...\n");
    
    quantum_config_t* config = quantum_config_create_default();
    assert(config != NULL);
    printf("[OK] Configuration par défaut créée\n");
    
    quantum_lum_t* qlum = quantum_lum_create(10, 20, 2);
    assert(qlum != NULL);
    assert(qlum->state_count == 2);
    printf("[OK] LUM Quantique créée (2 états)\n");
    
    bool gate_ok = quantum_apply_gate(qlum, QUANTUM_GATE_HADAMARD, config);
    assert(gate_ok == true);
    printf("[OK] Porte Hadamard appliquée avec succès\n");
    
    quantum_result_t* res = quantum_measure(qlum, config);
    assert(res != NULL);
    assert(res->success == true);
    printf("[OK] Mesure quantique effectuée (Collapse)\n");
    
    quantum_result_destroy(&res);
    quantum_lum_destroy(&qlum);
    quantum_config_destroy(&config);
    
    printf("[SUCCESS] Test unitaire Quantum Simulator terminé avec succès\n");
}

int main() {
    test_quantum_basic();
    return 0;
}

#include <stdio.h>
#include "../quantum_simulator_fusion_v2.h"
int main(void){
  quantum_fusion_v2_summary_t s;
  if(!quantum_fusion_v2_run_forensic_benchmark("results/fusion_forensic_full.jsonl",360,1400,QUANTUM_RNG_HARDWARE_PREFERRED,0xC0FFEEu,&s)){
    fprintf(stderr,"fusion benchmark failed\n");
    return 1;
  }
  printf("wins=%zu losses=%zu win_rate=%.9f nqubits_per_sec=%.3f hw_entropy=%s\n",
    s.nqubit_wins,s.baseline_wins,s.nqubit_win_rate,s.nqubits_per_sec,s.used_hardware_entropy?"true":"false");
  return 0;
}

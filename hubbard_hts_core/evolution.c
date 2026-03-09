#include <math.h>
#include <stddef.h>

double compute_energy(const double* psi, size_t n);
double measure_pairing(const double* psi, size_t n);

static void normalize_state(double* psi, size_t n) {
    double norm2 = 0.0;
    for (size_t i = 0; i < n; ++i) norm2 += psi[i] * psi[i];
    if (norm2 <= 1e-15) return;
    const double inv = 1.0 / sqrt(norm2);
    for (size_t i = 0; i < n; ++i) psi[i] *= inv;
}

void update_energy_pairing(double* psi, size_t n_sites, double* energy, double* pairing) {
    normalize_state(psi, n_sites);
    *energy = compute_energy(psi, n_sites);
    *pairing = measure_pairing(psi, n_sites);
    if (n_sites > 0) {
        *energy /= (double)n_sites;
        *pairing /= (double)n_sites;
    }
}

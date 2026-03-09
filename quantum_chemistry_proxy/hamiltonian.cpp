#include <vector>
#include <cmath>

double expectation_h(const std::vector<double>& state);
double measure_pairing(const std::vector<double>& state);

static void normalize_state(std::vector<double>& state) {
    double norm2 = 0.0;
    for (double v : state) norm2 += v * v;
    if (norm2 <= 1e-15) return;
    const double inv = 1.0 / std::sqrt(norm2);
    for (double& v : state) v *= inv;
}

void compute_step(std::vector<double>& state, double& energy, double& pairing) {
    normalize_state(state);
    energy = expectation_h(state);
    pairing = measure_pairing(state);
    if (!state.empty()) {
        energy /= static_cast<double>(state.size());
        pairing /= static_cast<double>(state.size());
    }
}

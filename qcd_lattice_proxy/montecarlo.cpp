#include <vector>
#include <cmath>

double compute_energy(const std::vector<double>& field);

struct Observable {
    double energy;
    double sign_ratio;
};

Observable measure_observable(const std::vector<double>& field,
                              const std::vector<double>& fermion_weights) {
    Observable out{};
    out.energy = compute_energy(field);
    if (!field.empty()) out.energy /= static_cast<double>(field.size());

    double sign_sum = 0.0;
    for (double w : fermion_weights) sign_sum += (w >= 0.0 ? 1.0 : -1.0);
    out.sign_ratio = fermion_weights.empty() ? 0.0 : sign_sum / static_cast<double>(fermion_weights.size());
    return out;
}

#pragma once

#include <string>
#include <vector>

namespace day019 {

using Distribution = std::vector<double>;

void validate_distribution(const Distribution& distribution,
                           const std::string& name);
double entropy_bits(const Distribution& p);
double cross_entropy_bits(const Distribution& p, const Distribution& q);
double kl_divergence_bits(const Distribution& p, const Distribution& q);
double js_divergence_bits(const Distribution& p, const Distribution& q);
double total_variation(const Distribution& p, const Distribution& q);
Distribution smoothed_binary(double probability_first, double epsilon);

}  // namespace day019

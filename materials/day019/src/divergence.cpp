#include "day019/divergence.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace day019 {
namespace {

constexpr double kSumTolerance = 1e-12;

void validate_pair(const Distribution& p, const Distribution& q) {
  validate_distribution(p, "p");
  validate_distribution(q, "q");
  if (p.size() != q.size()) {
    throw std::invalid_argument("p and q must have the same shape");
  }
}

}  // namespace

void validate_distribution(const Distribution& distribution,
                           const std::string& name) {
  if (distribution.empty()) {
    throw std::invalid_argument(name + " must be non-empty");
  }
  for (double value : distribution) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::invalid_argument(name + " must contain finite non-negative values");
    }
  }
  const double sum =
      std::accumulate(distribution.begin(), distribution.end(), 0.0);
  if (std::abs(sum - 1.0) > kSumTolerance) {
    throw std::invalid_argument(name + " must sum to one");
  }
}

double entropy_bits(const Distribution& p) {
  validate_distribution(p, "p");
  double result = 0.0;
  for (double probability : p) {
    if (probability > 0.0) {
      result -= probability * std::log2(probability);
    }
  }
  return result;
}

double cross_entropy_bits(const Distribution& p, const Distribution& q) {
  validate_pair(p, q);
  double result = 0.0;
  for (std::size_t index = 0; index < p.size(); ++index) {
    if (p[index] == 0.0) {
      continue;
    }
    if (q[index] == 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    result -= p[index] * std::log2(q[index]);
  }
  return result;
}

double kl_divergence_bits(const Distribution& p, const Distribution& q) {
  const double cross_entropy = cross_entropy_bits(p, q);
  if (std::isinf(cross_entropy)) {
    return cross_entropy;
  }
  return cross_entropy - entropy_bits(p);
}

double js_divergence_bits(const Distribution& p, const Distribution& q) {
  validate_pair(p, q);
  Distribution mixture(p.size(), 0.0);
  for (std::size_t index = 0; index < p.size(); ++index) {
    mixture[index] = 0.5 * (p[index] + q[index]);
  }
  return 0.5 * kl_divergence_bits(p, mixture) +
         0.5 * kl_divergence_bits(q, mixture);
}

double total_variation(const Distribution& p, const Distribution& q) {
  validate_pair(p, q);
  double l1_distance = 0.0;
  for (std::size_t index = 0; index < p.size(); ++index) {
    l1_distance += std::abs(p[index] - q[index]);
  }
  return 0.5 * l1_distance;
}

Distribution smoothed_binary(double probability_first, double epsilon) {
  if (!std::isfinite(probability_first) || probability_first < 0.0 ||
      probability_first > 1.0) {
    throw std::invalid_argument("probability_first must be in [0, 1]");
  }
  if (!std::isfinite(epsilon) || epsilon < 0.0) {
    throw std::invalid_argument("epsilon must be finite and non-negative");
  }
  const double denominator = 1.0 + 2.0 * epsilon;
  return {(probability_first + epsilon) / denominator,
          (1.0 - probability_first + epsilon) / denominator};
}

}  // namespace day019

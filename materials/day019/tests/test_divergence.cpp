#include "day019/divergence.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool close(double a, double b, double tolerance = 1e-12) {
  return std::abs(a - b) <= tolerance;
}

}  // namespace

int main() {
  try {
    const day019::Distribution p{0.5, 0.5};
    const day019::Distribution q{0.75, 0.25};
    check(close(day019::entropy_bits(p), 1.0), "entropy");
    check(close(day019::cross_entropy_bits(p, q), 1.207518749639422),
          "cross entropy");
    check(close(day019::kl_divergence_bits(p, q), 0.207518749639422),
          "KL");
    check(day019::kl_divergence_bits(p, q) !=
              day019::kl_divergence_bits(q, p),
          "KL must demonstrate asymmetry");
    check(close(day019::js_divergence_bits(p, q),
                day019::js_divergence_bits(q, p)),
          "JS symmetry");
    check(close(day019::total_variation(p, q), 0.25),
          "total variation");
    check(std::isinf(
              day019::kl_divergence_bits({1.0, 0.0}, {0.0, 1.0})),
          "support mismatch");
    check(close(day019::js_divergence_bits({1.0, 0.0}, {0.0, 1.0}),
                1.0),
          "disjoint JS");
    check(close(day019::smoothed_binary(1.0, 0.01)[0],
                1.01 / 1.02),
          "smoothing");
    std::cout << "CTEST_OK: 9 checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CTEST_ERROR: " << error.what() << '\n';
    return 1;
  }
}

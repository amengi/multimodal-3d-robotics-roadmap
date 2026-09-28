#include "day019/divergence.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(double actual, double expected, double tolerance,
                   const std::string& message) {
  require(std::abs(actual - expected) <= tolerance, message);
}

void write_scan(const std::string& path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open CSV path: " + path);
  }
  output << "epsilon,kl_pq_bits,js_bits\n";
  for (double epsilon : std::vector<double>{0.0, 1e-6, 1e-3, 1e-2, 5e-2}) {
    const auto p = day019::smoothed_binary(1.0, epsilon);
    const auto q = day019::smoothed_binary(0.0, epsilon);
    const double kl = day019::kl_divergence_bits(p, q);
    output << std::setprecision(12) << epsilon << ',';
    if (std::isinf(kl)) {
      output << "inf";
    } else {
      output << kl;
    }
    output << ',' << day019::js_divergence_bits(p, q) << '\n';
  }
}

void run_self_test() {
  using day019::Distribution;
  const Distribution p{0.5, 0.5};
  const Distribution q{0.75, 0.25};
  require_close(day019::entropy_bits(p), 1.0, 1e-12, "entropy");
  require_close(day019::cross_entropy_bits(p, q), 1.207518749639422,
                1e-12, "cross entropy");
  require_close(day019::kl_divergence_bits(p, q), 0.207518749639422,
                1e-12, "KL p||q");
  require_close(day019::kl_divergence_bits(q, p), 0.18872187554086717,
                1e-12, "KL q||p");
  require_close(day019::js_divergence_bits(p, q), 0.048794940695398636,
                1e-12, "JS");
  require_close(day019::total_variation(p, q), 0.25, 1e-12,
                "total variation");
  require_close(day019::js_divergence_bits(p, q),
                day019::js_divergence_bits(q, p), 1e-12, "JS symmetry");
  require_close(day019::cross_entropy_bits(p, q),
                day019::entropy_bits(p) +
                    day019::kl_divergence_bits(p, q),
                1e-12, "cross entropy identity");
  require(std::isinf(day019::kl_divergence_bits({1.0, 0.0}, {0.0, 1.0})),
          "support mismatch must be infinite");
  require_close(day019::js_divergence_bits({1.0, 0.0}, {0.0, 1.0}), 1.0,
                1e-12, "disjoint JS");
  bool rejected = false;
  try {
    day019::validate_distribution({0.4, 0.4}, "bad");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "invalid distribution must be rejected");
  std::cout << "SELF_TEST_OK: 11 checks\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      run_self_test();
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--csv") {
      write_scan(argv[2]);
      std::cout << "CSV_OK=" << argv[2] << '\n';
      return 0;
    }
    if (argc != 1) {
      std::cerr << "usage: divergence_demo [--self-test | --csv PATH]\n";
      return 2;
    }

    const day019::Distribution p{0.5, 0.5};
    const day019::Distribution q{0.75, 0.25};
    const double h_p = day019::entropy_bits(p);
    const double h_pq = day019::cross_entropy_bits(p, q);
    const double kl_pq = day019::kl_divergence_bits(p, q);
    const double kl_qp = day019::kl_divergence_bits(q, p);
    const double js = day019::js_divergence_bits(p, q);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "p_shape=(2) q_shape=(2) log_base=2 unit=bit\n";
    std::cout << "entropy_p_bits=" << h_p << '\n';
    std::cout << "cross_entropy_pq_bits=" << h_pq << '\n';
    std::cout << "kl_pq_bits=" << kl_pq << '\n';
    std::cout << "kl_qp_bits=" << kl_qp << '\n';
    std::cout << "kl_asymmetry_bits=" << std::abs(kl_pq - kl_qp) << '\n';
    std::cout << "identity_error_bits=" << std::abs(h_pq - h_p - kl_pq)
              << '\n';
    std::cout << "js_bits=" << js << '\n';
    std::cout << "total_variation=" << day019::total_variation(p, q) << '\n';
    std::cout << "support_mismatch_kl_bits="
              << day019::kl_divergence_bits({1.0, 0.0}, {0.0, 1.0})
              << '\n';
    const auto p_smooth = day019::smoothed_binary(1.0, 0.01);
    const auto q_smooth = day019::smoothed_binary(0.0, 0.01);
    std::cout << "smoothed_epsilon=0.010000 smoothed_kl_bits="
              << day019::kl_divergence_bits(p_smooth, q_smooth)
              << " smoothed_js_bits="
              << day019::js_divergence_bits(p_smooth, q_smooth) << '\n';
    std::cout << "SUCCESS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

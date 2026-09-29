#include "day020/analysis.hpp"

#include <Eigen/Dense>

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
    Eigen::Matrix<double, 5, 2> design;
    design << 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0;
    Eigen::Matrix<double, 5, 1> y;
    y << 1.1, 2.9, 5.2, 6.8, 9.1;
    const auto fit = day020::fit_line(design, y);
    check(close(fit.qr_beta(0), 1.99), "QR slope");
    check(close(fit.qr_beta(1), 1.04), "QR intercept");
    check((fit.qr_beta - fit.svd_beta).norm() < 1e-12, "QR versus SVD");
    check(close(fit.rmse_m, 0.1462873883832781), "RMSE");
    check(close(fit.condition_number, 4.73872001868727), "condition number");

    const auto paired =
        day020::analyze_counts((Eigen::Matrix2d() << 3, 1, 1, 3).finished());
    check(close(paired.h_x_bits, 1.0), "H(X)");
    check(close(paired.h_joint_bits, 1.811278124459133), "H(X,Y)");
    check(close(paired.h_x_given_y_bits, 0.8112781244591328), "H(X|Y)");
    check(close(paired.mi_from_conditional_bits, 0.18872187554086717),
          "MI conditional");
    check(close(paired.mi_from_conditional_bits, paired.mi_from_kl_bits),
          "MI equality");
    check(close(paired.match_rule_accuracy, 0.75), "task accuracy");
    check(close(paired.majority_y_accuracy, 0.5), "majority baseline");

    const auto shifted =
        day020::analyze_counts((Eigen::Matrix2d() << 2, 2, 2, 2).finished());
    check(close(shifted.mi_from_kl_bits, 0.0), "independent MI");
    check(close(shifted.match_rule_accuracy, 0.5), "shifted accuracy");
    std::cout << "CTEST_OK: 14 checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CTEST_ERROR: " << error.what() << '\n';
    return 1;
  }
}

#include "day020/analysis.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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

Eigen::Matrix<double, 5, 2> design_matrix() {
  Eigen::Matrix<double, 5, 2> design;
  design << 0.0, 1.0,
            1.0, 1.0,
            2.0, 1.0,
            3.0, 1.0,
            4.0, 1.0;
  return design;
}

Eigen::Matrix<double, 5, 1> observations() {
  Eigen::Matrix<double, 5, 1> values;
  values << 1.1, 2.9, 5.2, 6.8, 9.1;
  return values;
}

double three_way_error(const day020::MutualInformationResult& result) {
  return std::max({std::abs(result.mi_from_conditional_bits -
                            result.mi_from_entropies_bits),
                   std::abs(result.mi_from_conditional_bits -
                            result.mi_from_kl_bits),
                   std::abs(result.mi_from_entropies_bits -
                            result.mi_from_kl_bits)});
}

void write_csv(const std::string& path) {
  const auto paired = day020::analyze_counts((Eigen::Matrix2d() << 3, 1, 1, 3).finished());
  const auto shifted = day020::analyze_counts((Eigen::Matrix2d() << 2, 2, 2, 2).finished());
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open CSV path: " + path);
  }
  output << "case,h_x_bits,h_y_bits,h_joint_bits,h_x_given_y_bits,"
            "mi_conditional_bits,mi_entropies_bits,mi_kl_bits,"
            "match_accuracy,majority_y_accuracy\n";
  output << std::setprecision(12);
  const auto write_row = [&output](const std::string& name,
                                    const day020::MutualInformationResult& r) {
    output << name << ',' << r.h_x_bits << ',' << r.h_y_bits << ','
           << r.h_joint_bits << ',' << r.h_x_given_y_bits << ','
           << r.mi_from_conditional_bits << ',' << r.mi_from_entropies_bits
           << ',' << r.mi_from_kl_bits << ',' << r.match_rule_accuracy << ','
           << r.majority_y_accuracy << '\n';
  };
  write_row("paired", paired);
  write_row("shift_y_by_2", shifted);
}

void run_self_test() {
  const auto design = design_matrix();
  const auto top_block = design.block<3, 2>(0, 0);
  require(top_block.rows() == 3 && top_block.cols() == 2,
          "fixed-size block shape");
  require_close(top_block.sum(), 6.0, 1e-12, "fixed-size block values");
  const auto fit = day020::fit_line(design, observations());
  require_close(fit.qr_beta(0), 1.99, 1e-12, "QR slope");
  require_close(fit.qr_beta(1), 1.04, 1e-12, "QR intercept");
  require((fit.qr_beta - fit.svd_beta).cwiseAbs().maxCoeff() < 1e-12,
          "QR and SVD solutions");
  require_close(fit.rmse_m, 0.1462873883832781, 1e-12, "RMSE");
  require_close(fit.singular_values(0), 5.788593144588945, 1e-12,
                "largest singular value");
  require_close(fit.singular_values(1), 1.221552048182098, 1e-12,
                "smallest singular value");
  require_close(fit.condition_number, 4.73872001868727, 1e-12,
                "condition number");

  const auto paired = day020::analyze_counts((Eigen::Matrix2d() << 3, 1, 1, 3).finished());
  require_close(paired.h_x_bits, 1.0, 1e-12, "H(X)");
  require_close(paired.h_y_bits, 1.0, 1e-12, "H(Y)");
  require_close(paired.h_joint_bits, 1.811278124459133, 1e-12,
                "H(X,Y)");
  require_close(paired.h_x_given_y_bits, 0.8112781244591328, 1e-12,
                "H(X|Y)");
  require_close(paired.mi_from_conditional_bits, 0.18872187554086717,
                1e-12, "MI");
  require(three_way_error(paired) < 1e-12, "three MI definitions");
  require_close(paired.match_rule_accuracy, 0.75, 1e-12,
                "match-rule accuracy");
  require_close(paired.majority_y_accuracy, 0.5, 1e-12,
                "majority baseline");

  const auto shifted = day020::analyze_counts((Eigen::Matrix2d() << 2, 2, 2, 2).finished());
  require_close(shifted.mi_from_kl_bits, 0.0, 1e-12, "independent MI");
  require_close(shifted.match_rule_accuracy, 0.5, 1e-12,
                "shifted match accuracy");
  std::cout << "SELF_TEST_OK: 19 checks\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      run_self_test();
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--csv") {
      write_csv(argv[2]);
      std::cout << "CSV_OK=" << argv[2] << '\n';
      return 0;
    }
    if (argc != 1) {
      std::cerr << "usage: eigen_mi_demo [--self-test | --csv PATH]\n";
      return 2;
    }

    const auto design = design_matrix();
    const auto top_block = design.block<3, 2>(0, 0);
    const auto fit = day020::fit_line(design, observations());
    const auto paired = day020::analyze_counts((Eigen::Matrix2d() << 3, 1, 1, 3).finished());
    const auto shifted = day020::analyze_counts((Eigen::Matrix2d() << 2, 2, 2, 2).finished());

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "eigen_version=" << EIGEN_WORLD_VERSION << '.'
              << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION << '\n';
    std::cout << "design_shape=(5,2) observation_shape=(5,) unit=m\n";
    std::cout << "top_block_shape=(3,2) top_block_sum=" << top_block.sum()
              << '\n';
    std::cout << "qr_slope=" << fit.qr_beta(0) << '\n';
    std::cout << "qr_intercept_m=" << fit.qr_beta(1) << '\n';
    std::cout << "svd_slope=" << fit.svd_beta(0) << '\n';
    std::cout << "svd_intercept_m=" << fit.svd_beta(1) << '\n';
    std::cout << "beta_max_abs_diff="
              << (fit.qr_beta - fit.svd_beta).cwiseAbs().maxCoeff() << '\n';
    std::cout << "rmse_m=" << fit.rmse_m << '\n';
    std::cout << "singular_values=" << fit.singular_values(0) << ','
              << fit.singular_values(1) << '\n';
    std::cout << "condition_number=" << fit.condition_number << '\n';

    std::cout << "joint_counts=3,1,1,3 total=8 log_base=2 unit=bit\n";
    std::cout << "p_x=" << paired.p_x(0) << ',' << paired.p_x(1) << '\n';
    std::cout << "p_y=" << paired.p_y(0) << ',' << paired.p_y(1) << '\n';
    std::cout << "h_x_bits=" << paired.h_x_bits << '\n';
    std::cout << "h_y_bits=" << paired.h_y_bits << '\n';
    std::cout << "h_joint_bits=" << paired.h_joint_bits << '\n';
    std::cout << "h_x_given_y_bits=" << paired.h_x_given_y_bits << '\n';
    std::cout << "h_y_given_x_bits=" << paired.h_y_given_x_bits << '\n';
    std::cout << "mi_conditional_bits=" << paired.mi_from_conditional_bits
              << '\n';
    std::cout << "mi_entropies_bits=" << paired.mi_from_entropies_bits << '\n';
    std::cout << "mi_kl_bits=" << paired.mi_from_kl_bits << '\n';
    std::cout << "mi_three_way_error=" << three_way_error(paired) << '\n';
    std::cout << "match_rule_accuracy=" << paired.match_rule_accuracy << '\n';
    std::cout << "majority_y_accuracy=" << paired.majority_y_accuracy << '\n';
    std::cout << "shifted_mi_bits=" << shifted.mi_from_kl_bits << '\n';
    std::cout << "shifted_match_accuracy=" << shifted.match_rule_accuracy << '\n';
    std::cout << "SUCCESS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

#pragma once

#include <Eigen/Dense>

namespace day020 {

struct LineFitResult {
  Eigen::Vector2d qr_beta;
  Eigen::Vector2d svd_beta;
  Eigen::Vector2d singular_values;
  double rmse_m;
  double condition_number;
};

struct MutualInformationResult {
  Eigen::Matrix2d probabilities;
  Eigen::Vector2d p_x;
  Eigen::Vector2d p_y;
  double h_x_bits;
  double h_y_bits;
  double h_joint_bits;
  double h_x_given_y_bits;
  double h_y_given_x_bits;
  double mi_from_conditional_bits;
  double mi_from_entropies_bits;
  double mi_from_kl_bits;
  double match_rule_accuracy;
  double majority_y_accuracy;
};

LineFitResult fit_line(const Eigen::MatrixXd& design,
                       const Eigen::VectorXd& observations_m);
MutualInformationResult analyze_counts(const Eigen::Matrix2d& counts);

}  // namespace day020

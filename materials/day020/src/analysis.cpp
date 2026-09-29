#include "day020/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace day020 {
namespace {

template <typename Derived>
double entropy_bits(const Eigen::MatrixBase<Derived>& probabilities) {
  double entropy = 0.0;
  for (Eigen::Index index = 0; index < probabilities.size(); ++index) {
    const double probability = probabilities(index);
    if (probability > 0.0) {
      entropy -= probability * std::log2(probability);
    }
  }
  return entropy;
}

template <typename Derived>
void require_finite(const Eigen::MatrixBase<Derived>& values, const char* name) {
  if (!values.allFinite()) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

}  // namespace

LineFitResult fit_line(const Eigen::MatrixXd& design,
                       const Eigen::VectorXd& observations_m) {
  require_finite(design, "design");
  require_finite(observations_m, "observations");
  if (design.rows() != observations_m.size()) {
    throw std::invalid_argument("design rows must match observation count");
  }
  if (design.rows() <= design.cols() || design.cols() != 2) {
    throw std::invalid_argument("expected an overdetermined (N,2) design");
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
  if (qr.rank() != design.cols()) {
    throw std::invalid_argument("design matrix must have full column rank");
  }
  const Eigen::Vector2d qr_beta = qr.solve(observations_m);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      design, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::Vector2d svd_beta = svd.solve(observations_m);
  const Eigen::Vector2d singular_values = svd.singularValues();
  if (singular_values(1) <= std::numeric_limits<double>::epsilon()) {
    throw std::invalid_argument("smallest singular value is numerically zero");
  }

  const Eigen::VectorXd residual_m = design * svd_beta - observations_m;
  const double rmse_m = std::sqrt(residual_m.squaredNorm() / design.rows());
  const double condition_number = singular_values(0) / singular_values(1);
  return {qr_beta, svd_beta, singular_values, rmse_m, condition_number};
}

MutualInformationResult analyze_counts(const Eigen::Matrix2d& counts) {
  require_finite(counts, "counts");
  if ((counts.array() < 0.0).any()) {
    throw std::invalid_argument("counts must be non-negative");
  }
  const double total = counts.sum();
  if (total <= 0.0) {
    throw std::invalid_argument("counts must have positive total");
  }

  const Eigen::Matrix2d probabilities = counts / total;
  const Eigen::Vector2d p_x = probabilities.rowwise().sum();
  const Eigen::Vector2d p_y = probabilities.colwise().sum().transpose();
  const double h_x = entropy_bits(p_x);
  const double h_y = entropy_bits(p_y);
  const Eigen::Map<const Eigen::VectorXd> joint_vector(probabilities.data(), 4);
  const double h_joint = entropy_bits(joint_vector);
  const double h_x_given_y = h_joint - h_y;
  const double h_y_given_x = h_joint - h_x;
  const double mi_from_conditional = h_x - h_x_given_y;
  const double mi_from_entropies = h_x + h_y - h_joint;

  double mi_from_kl = 0.0;
  for (Eigen::Index x = 0; x < 2; ++x) {
    for (Eigen::Index y = 0; y < 2; ++y) {
      const double joint = probabilities(x, y);
      if (joint > 0.0) {
        mi_from_kl += joint * std::log2(joint / (p_x(x) * p_y(y)));
      }
    }
  }

  const double match_rule_accuracy = (counts(0, 0) + counts(1, 1)) / total;
  const double majority_y_accuracy = std::max(p_y(0), p_y(1));
  return {probabilities,
          p_x,
          p_y,
          h_x,
          h_y,
          h_joint,
          h_x_given_y,
          h_y_given_x,
          mi_from_conditional,
          mi_from_entropies,
          mi_from_kl,
          match_rule_accuracy,
          majority_y_accuracy};
}

}  // namespace day020

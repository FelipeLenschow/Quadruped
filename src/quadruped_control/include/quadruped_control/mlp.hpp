#pragma once

#include <vector>

#include <Eigen/Dense>

namespace quadruped_control {

using VecXf = Eigen::VectorXf;
using MatXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Actor of every policy here: (x - mean) / sqrt(var + 1e-8), then Linear+ELU layers, then a Linear head.
class Mlp {
 public:
  struct Layer {
    MatXf weight;  // [out, in]
    VecXf bias;
  };

  Mlp() = default;
  Mlp(VecXf mean, VecXf var, std::vector<Layer> layers);

  VecXf forward(const VecXf& obs) const;

  int input_dim() const { return static_cast<int>(mean_.size()); }
  int output_dim() const { return layers_.empty() ? 0 : static_cast<int>(layers_.back().bias.size()); }

 private:
  VecXf mean_;
  VecXf std_;
  std::vector<Layer> layers_;
};

}  // namespace quadruped_control

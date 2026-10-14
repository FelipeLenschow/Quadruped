#include "quadruped_control/mlp.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace quadruped_control {

Mlp::Mlp(VecXf mean, VecXf var, std::vector<Layer> layers)
    : mean_(std::move(mean)), layers_(std::move(layers)) {
  if (layers_.empty()) throw std::invalid_argument("[Mlp] No layers.");
  std_ = (var.array() + 1e-8f).sqrt();
  Eigen::Index width = mean_.size();
  for (size_t i = 0; i < layers_.size(); ++i) {
    const Layer& l = layers_[i];
    if (l.weight.cols() != width || l.bias.size() != l.weight.rows()) {
      throw std::invalid_argument("[Mlp] Layer " + std::to_string(i) + " does not fit the one before it.");
    }
    width = l.weight.rows();
  }
}

VecXf Mlp::forward(const VecXf& obs) const {
  VecXf x = (obs - mean_).cwiseQuotient(std_);
  const size_t hidden = layers_.size() - 1;
  for (size_t i = 0; i < hidden; ++i) {
    x = layers_[i].weight * x + layers_[i].bias;
    x = x.unaryExpr([](float v) { return v > 0.0f ? v : std::expm1(v); });
  }
  return layers_.back().weight * x + layers_.back().bias;
}

}  // namespace quadruped_control

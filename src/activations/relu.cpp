#include "activations/relu.h"

namespace mnist {

Eigen::MatrixXf ReLU::forward(const Eigen::MatrixXf& x) {
    mask_ = (x.array() > 0.0f).cast<float>();
    return x.cwiseMax(0.0f);
}

Eigen::MatrixXf ReLU::backward(const Eigen::MatrixXf& grad) {
    return grad.array() * mask_.array();
}

}  // namespace mnist

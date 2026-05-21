#include "layers/flatten.h"

namespace mnist {

Eigen::MatrixXf Flatten::forward(const Eigen::MatrixXf& x) {
    // With batch format, input is already (C*H*W, B) — no reshape needed
    return x;
}

Eigen::MatrixXf Flatten::backward(const Eigen::MatrixXf& grad) {
    return grad;
}

}  // namespace mnist

#include "activations/softmax.h"

namespace mnist {

Eigen::MatrixXf Softmax::forward(const Eigen::MatrixXf& x) {
    // Numerically stable softmax: subtract max before exp
    float shift = x.maxCoeff();
    Eigen::MatrixXf exp_x = (x.array() - shift).exp();
    probs_ = exp_x / exp_x.sum();
    return probs_;
}

Eigen::MatrixXf Softmax::backward(const Eigen::MatrixXf& grad) {
    // grad = dL/dp  (gradient of loss w.r.t. softmax output probs_)
    // Returns dL/dx where x is the input to softmax
    // Jacobian of softmax: diag(p) - p*p^T
    Eigen::MatrixXf dx = probs_.array() * grad.array();
    dx.array() -= probs_.array() * (probs_.transpose() * grad).value();
    return dx;
}

}  // namespace mnist

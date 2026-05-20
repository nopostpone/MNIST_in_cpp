#include "layers/fully_connected.h"

namespace mnist {

Linear::Linear(int in_features, int out_features)
    : W_(out_features, in_features), b_(out_features, 1)
{
    static std::mt19937 rng(42);
    he_init(W_.data, in_features, rng);
    W_.grad.setZero();
    b_.data.setZero();
    b_.grad.setZero();
}

Eigen::MatrixXf Linear::forward(const Eigen::MatrixXf& x) {
    x_cache_ = x;  // (in_features, 1)
    return W_.data * x + b_.data;  // (out_features, 1)
}

Eigen::MatrixXf Linear::backward(const Eigen::MatrixXf& grad) {
    // grad: (out_features, 1)
    W_.grad += grad * x_cache_.transpose();  // (out, 1) x (1, in) = (out, in)
    b_.grad += grad;                          // (out, 1)
    return W_.data.transpose() * grad;        // (in, 1)
}

}  // namespace mnist

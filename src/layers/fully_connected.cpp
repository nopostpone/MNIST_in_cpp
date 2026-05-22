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
    x_cache_ = x;
    Eigen::MatrixXf out = W_.data * x;
    for (int r = 0; r < out.rows(); ++r)
        out.row(r).array() += b_.data(r, 0);
    return out;
}

Eigen::MatrixXf Linear::backward(const Eigen::MatrixXf& grad) {
    W_.grad += grad * x_cache_.transpose();
    for (int r = 0; r < grad.rows(); ++r)
        b_.grad(r, 0) += grad.row(r).sum();
    return W_.data.transpose() * grad;
}

}  // namespace mnist

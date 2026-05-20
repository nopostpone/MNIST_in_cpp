#include "layers/flatten.h"

namespace mnist {

Eigen::MatrixXf Flatten::forward(const Eigen::MatrixXf& x) {
    // x: (C, H*W) → out: (C*H*W, 1)
    int total = static_cast<int>(x.size());
    Eigen::MatrixXf out(total, 1);
    int idx = 0;
    for (int r = 0; r < x.rows(); ++r)
        for (int c = 0; c < x.cols(); ++c)
            out(idx++, 0) = x(r, c);
    return out;
}

Eigen::MatrixXf Flatten::backward(const Eigen::MatrixXf& grad) {
    // grad: (C*H*W, 1) → out: (C, H*W)
    int cols = in_h_ * in_w_;
    Eigen::MatrixXf dx(in_c_, cols);
    int idx = 0;
    for (int r = 0; r < in_c_; ++r)
        for (int c = 0; c < cols; ++c)
            dx(r, c) = grad(idx++, 0);
    return dx;
}

}  // namespace mnist

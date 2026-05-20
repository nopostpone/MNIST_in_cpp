#include "layers/conv2d.h"

namespace mnist {

Conv2D::Conv2D(int in_channels, int out_channels,
               int kernel_size, int stride, int padding,
               int input_h, int input_w)
    : in_c_(in_channels), out_c_(out_channels),
      h_(input_h), w_(input_w),
      k_(kernel_size), s_(stride), p_(padding)
{
    h_out_ = (h_ - k_ + 2 * p_) / s_ + 1;
    w_out_ = (w_ - k_ + 2 * p_) / s_ + 1;

    int fan_in = in_c_ * k_ * k_;
    W_ = Parameter(out_c_, fan_in);
    b_ = Parameter(out_c_, 1);

    static std::mt19937 rng(42);
    he_init(W_.data, fan_in, rng);
    b_.data.setZero();
}

Eigen::MatrixXf Conv2D::im2col(const Eigen::MatrixXf& x) {
    // x: (in_c_, h_*w_) → cols: (in_c_*k_*k_, h_out_*w_out_)
    int patch_sz = in_c_ * k_ * k_;
    int n_patches = h_out_ * w_out_;
    Eigen::MatrixXf cols(patch_sz, n_patches);

    int col = 0;
    for (int i = 0; i < h_out_; ++i) {
        for (int j = 0; j < w_out_; ++j) {
            int row = 0;
            for (int c = 0; c < in_c_; ++c) {
                for (int ki = 0; ki < k_; ++ki) {
                    for (int kj = 0; kj < k_; ++kj) {
                        int hi = i * s_ + ki - p_;
                        int wi = j * s_ + kj - p_;
                        if (hi >= 0 && hi < h_ && wi >= 0 && wi < w_)
                            cols(row, col) = x(c, hi * w_ + wi);
                        else
                            cols(row, col) = 0.0f;
                        ++row;
                    }
                }
            }
            ++col;
        }
    }
    return cols;
}

Eigen::MatrixXf Conv2D::col2im(const Eigen::MatrixXf& dcols) {
    // dcols: (in_c_*k_*k_, h_out_*w_out_) → out: (in_c_, h_*w_)
    Eigen::MatrixXf dx(in_c_, h_ * w_);
    dx.setZero();

    int col = 0;
    for (int i = 0; i < h_out_; ++i) {
        for (int j = 0; j < w_out_; ++j) {
            int row = 0;
            for (int c = 0; c < in_c_; ++c) {
                for (int ki = 0; ki < k_; ++ki) {
                    for (int kj = 0; kj < k_; ++kj) {
                        int hi = i * s_ + ki - p_;
                        int wi = j * s_ + kj - p_;
                        if (hi >= 0 && hi < h_ && wi >= 0 && wi < w_)
                            dx(c, hi * w_ + wi) += dcols(row, col);
                        ++row;
                    }
                }
            }
            ++col;
        }
    }
    return dx;
}

Eigen::MatrixXf Conv2D::forward(const Eigen::MatrixXf& x) {
    // x: (in_c_, h_*w_) → out: (out_c_, h_out_*w_out_)
    col_cache_ = im2col(x);               // (in_c_*k_*k_, n_patches)
    Eigen::MatrixXf out = W_.data * col_cache_;  // (out_c_, n_patches)
    for (int oc = 0; oc < out_c_; ++oc)
        out.row(oc).array() += b_.data(oc, 0);
    return out;
}

Eigen::MatrixXf Conv2D::backward(const Eigen::MatrixXf& dout) {
    // dout: (out_c_, h_out_*w_out_)
    W_.grad += dout * col_cache_.transpose();  // (out_c_, in_c_*k_*k_)
    for (int oc = 0; oc < out_c_; ++oc)
        b_.grad(oc, 0) += dout.row(oc).sum();

    Eigen::MatrixXf dcols = W_.data.transpose() * dout;  // (fan_in, n_patches)
    return col2im(dcols);  // (in_c_, h_*w_)
}

}  // namespace mnist

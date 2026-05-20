#include "layers/max_pool.h"
#include <limits>

namespace mnist {

MaxPool2D::MaxPool2D(int channels, int input_h, int input_w,
                     int pool_size, int stride)
    : c_(channels), h_in_(input_h), w_in_(input_w),
      pool_(pool_size), stride_(stride)
{
    h_out_ = (h_in_ - pool_) / stride_ + 1;
    w_out_ = (w_in_ - pool_) / stride_ + 1;
}

Eigen::MatrixXf MaxPool2D::forward(const Eigen::MatrixXf& x) {
    // x: (c_, h_in_*w_in_) → out: (c_, h_out_*w_out_)
    Eigen::MatrixXf out(c_, h_out_ * w_out_);
    mask_.resize(c_, h_out_ * w_out_);

    for (int ch = 0; ch < c_; ++ch) {
        for (int i = 0; i < h_out_; ++i) {
            for (int j = 0; j < w_out_; ++j) {
                float best = -std::numeric_limits<float>::infinity();
                int best_idx = -1;
                for (int pi = 0; pi < pool_; ++pi) {
                    for (int pj = 0; pj < pool_; ++pj) {
                        int hi = i * stride_ + pi;
                        int wi = j * stride_ + pj;
                        float val = x(ch, hi * w_in_ + wi);
                        if (val > best) {
                            best = val;
                            best_idx = hi * w_in_ + wi;
                        }
                    }
                }
                out(ch, i * w_out_ + j) = best;
                mask_(ch, i * w_out_ + j) = best_idx;
            }
        }
    }
    return out;
}

Eigen::MatrixXf MaxPool2D::backward(const Eigen::MatrixXf& dout) {
    // dout: (c_, h_out_*w_out_) → dx: (c_, h_in_*w_in_)
    Eigen::MatrixXf dx(c_, h_in_ * w_in_);
    dx.setZero();

    for (int ch = 0; ch < c_; ++ch) {
        for (int i = 0; i < h_out_; ++i) {
            for (int j = 0; j < w_out_; ++j) {
                int idx = mask_(ch, i * w_out_ + j);
                dx(ch, idx) += dout(ch, i * w_out_ + j);
            }
        }
    }
    return dx;
}

}  // namespace mnist

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
    // x: (c_ * h_in_ * w_in_, B) → out: (c_ * h_out_ * w_out_, B)
    int B = static_cast<int>(x.cols());
    int in_per_sample = c_ * h_in_ * w_in_;
    int out_per_sample = c_ * h_out_ * w_out_;

    Eigen::MatrixXf out(out_per_sample, B);
    mask_.resize(out_per_sample, B);

    for (int b = 0; b < B; ++b) {
        for (int ch = 0; ch < c_; ++ch) {
            int ch_off = ch * h_in_ * w_in_;
            int out_ch_off = ch * h_out_ * w_out_;
            for (int i = 0; i < h_out_; ++i) {
                for (int j = 0; j < w_out_; ++j) {
                    float best = -std::numeric_limits<float>::infinity();
                    int best_idx = -1;
                    for (int pi = 0; pi < pool_; ++pi) {
                        for (int pj = 0; pj < pool_; ++pj) {
                            int hi = i * stride_ + pi;
                            int wi = j * stride_ + pj;
                            int idx = ch_off + hi * w_in_ + wi;
                            float val = x(idx, b);
                            if (val > best) {
                                best = val;
                                best_idx = idx;
                            }
                        }
                    }
                    int out_idx = out_ch_off + i * w_out_ + j;
                    out(out_idx, b) = best;
                    mask_(out_idx, b) = best_idx;
                }
            }
        }
    }
    return out;
}

Eigen::MatrixXf MaxPool2D::backward(const Eigen::MatrixXf& dout) {
    // dout: (c_ * h_out_ * w_out_, B) → dx: (c_ * h_in_ * w_in_, B)
    int B = static_cast<int>(dout.cols());
    int in_per_sample = c_ * h_in_ * w_in_;

    Eigen::MatrixXf dx(in_per_sample, B);
    dx.setZero();

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < c_ * h_out_ * w_out_; ++i) {
            int idx = mask_(i, b);
            dx(idx, b) += dout(i, b);
        }
    }
    return dx;
}

}  // namespace mnist

#pragma once
#include "core/module.h"

namespace mnist {

class Conv2D : public Module {
    int in_c_, out_c_, h_, w_, k_, s_, p_;
    int h_out_, w_out_;
    Parameter W_;  // (out_c_, in_c_ * k_ * k_)
    Parameter b_;  // (out_c_, 1)
    Eigen::MatrixXf col_cache_;

    Eigen::MatrixXf im2col(const Eigen::MatrixXf& x);
    void col2im_flat(const Eigen::MatrixXf& dcols, float* dsrc);
public:
    Conv2D(int in_channels, int out_channels,
           int kernel_size, int stride, int padding,
           int input_h, int input_w);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override;
    std::vector<Parameter*> parameters() override { return {&W_, &b_}; }
};

}  // namespace mnist

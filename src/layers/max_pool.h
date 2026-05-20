#pragma once
#include "core/module.h"

namespace mnist {

class MaxPool2D : public Module {
    int c_, h_in_, w_in_, pool_, stride_;
    int h_out_, w_out_;
    Eigen::MatrixXi mask_;  // flat index of max within each window for backward
public:
    MaxPool2D(int channels, int input_h, int input_w,
              int pool_size, int stride);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override;
};

}  // namespace mnist

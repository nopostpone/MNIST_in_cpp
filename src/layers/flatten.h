#pragma once
#include "core/module.h"

namespace mnist {

class Flatten : public Module {
    int in_c_, in_h_, in_w_;
public:
    Flatten(int c, int h, int w) : in_c_(c), in_h_(h), in_w_(w) {}
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override;
};

}  // namespace mnist

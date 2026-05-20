#pragma once
#include "core/module.h"

namespace mnist {

class Linear : public Module {
    Parameter W_;
    Parameter b_;
    Eigen::MatrixXf x_cache_;
public:
    Linear(int in_features, int out_features);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override;
    std::vector<Parameter*> parameters() override { return {&W_, &b_}; }
};

}  // namespace mnist

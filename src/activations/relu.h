#pragma once
#include "core/module.h"

namespace mnist {

class ReLU : public Module {
    Eigen::MatrixXf mask_;
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override;
};

}  // namespace mnist

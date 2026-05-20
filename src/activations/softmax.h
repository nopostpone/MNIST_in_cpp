#pragma once
#include "core/module.h"

namespace mnist {

class Softmax : public Module {
    Eigen::MatrixXf probs_;
public:
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override;
};

}  // namespace mnist

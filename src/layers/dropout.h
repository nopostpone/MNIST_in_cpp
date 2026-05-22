#pragma once
#include "core/module.h"
#include <random>

namespace mnist {

class Dropout : public Module {
    float p_;                     // drop probability
    Eigen::MatrixXf mask_;        // cached mask for backward
    bool training_ = true;
public:
    explicit Dropout(float p) : p_(p) {}

    void set_training(bool t) { training_ = t; }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) override {
        if (!training_ || p_ <= 0.0f) return x;
        // Generate mask: Bernoulli(1-p), scale by 1/(1-p)
        mask_.resize(x.rows(), x.cols());
        static std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(1.0f - p_);
        float scale = 1.0f / (1.0f - p_);
        mask_ = Eigen::MatrixXf::NullaryExpr(x.rows(), x.cols(),
            [&]() { return dist(rng) ? scale : 0.0f; });
        return x.array() * mask_.array();
    }

    Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) override {
        if (!training_ || p_ <= 0.0f) return grad;
        return grad.array() * mask_.array();
    }
};

}  // namespace mnist

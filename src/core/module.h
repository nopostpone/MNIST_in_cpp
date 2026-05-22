#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace mnist {

struct Parameter {
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;

    Parameter() = default;
    Parameter(int rows, int cols) : data(rows, cols), grad(rows, cols) {
        data.setZero();
        grad.setZero();
    }
};

class Module {
public:
    virtual ~Module() = default;
    virtual Eigen::MatrixXf forward(const Eigen::MatrixXf& x) = 0;
    virtual Eigen::MatrixXf backward(const Eigen::MatrixXf& grad) = 0;
    virtual std::vector<Parameter*> parameters() { return {}; }

};

// Kaiming He initialization (for ReLU)
inline void he_init(Eigen::MatrixXf& w, int fan_in, std::mt19937& rng) {
    float std = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std);
    w = Eigen::MatrixXf::NullaryExpr(w.rows(), w.cols(),
        [&]() { return dist(rng); });
}

}  // namespace mnist

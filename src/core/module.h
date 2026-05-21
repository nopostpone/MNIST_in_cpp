#pragma once

#include <Eigen/Dense>
#include <random>
#include <vector>
#include <cstdio>
#include <iostream>

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

    void zero_grad() {
        for (auto* p : parameters()) p->grad.setZero();
    }
    void update(float lr) {
        for (auto* p : parameters()) p->data -= lr * p->grad;
    }
};

// Kaiming He 初始化 (适用于 ReLU)
inline void he_init(Eigen::MatrixXf& w, int fan_in, std::mt19937& rng) {
    float std = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std);
    w = Eigen::MatrixXf::NullaryExpr(w.rows(), w.cols(),
        [&]() { return dist(rng); });
}

// Save / load all parameters to a binary file (C FILE* to avoid GCC 15 O3 bug)
inline void save_weights(const std::vector<Parameter*>& params,
                         const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; return; }
    int n = static_cast<int>(params.size());
    std::fwrite(&n, sizeof(int), 1, f);
    for (auto* p : params) {
        int r = static_cast<int>(p->data.rows());
        int c = static_cast<int>(p->data.cols());
        std::fwrite(&r, sizeof(int), 1, f);
        std::fwrite(&c, sizeof(int), 1, f);
        std::fwrite(p->data.data(), sizeof(float), r * c, f);
    }
    std::fclose(f);
}

inline void load_weights(const std::vector<Parameter*>& params,
                         const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; return; }
    int n = 0;
    std::fread(&n, sizeof(int), 1, f);
    for (int i = 0; i < n; ++i) {
        int r = 0, c = 0;
        std::fread(&r, sizeof(int), 1, f);
        std::fread(&c, sizeof(int), 1, f);
        if (i < static_cast<int>(params.size())) {
            params[i]->data.resize(r, c);
            std::fread(params[i]->data.data(), sizeof(float), r * c, f);
        } else {
            std::fseek(f, r * c * sizeof(float), SEEK_CUR);
        }
    }
    std::fclose(f);
}

}  // namespace mnist

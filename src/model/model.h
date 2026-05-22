#pragma once
#include "core/module.h"
#include <memory>
#include <vector>

namespace mnist {

/// Sequential container — chains layers in order, like nn.Sequential
class Sequential {
    std::vector<std::unique_ptr<Module>> layers_;
public:
    Sequential() = default;

    template<typename T, typename... Args>
    T& add(Args&&... args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        T& ref = *ptr;
        layers_.push_back(std::move(ptr));
        return ref;
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);
    void backward(const Eigen::MatrixXf& grad);

    std::vector<Parameter*> parameters();
};

}  // namespace mnist

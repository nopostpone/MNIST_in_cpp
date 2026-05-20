#include "loss/cross_entropy.h"
#include <cmath>

namespace mnist {

float CrossEntropyLoss::forward(const Eigen::MatrixXf& pred, int label) {
    return -std::log(pred(label, 0) + 1e-7f);
}

Eigen::MatrixXf CrossEntropyLoss::backward(const Eigen::MatrixXf& pred, int label) {
    // Combined dL/d(logits) = softmax_output - one_hot(label)
    Eigen::MatrixXf grad = pred;
    grad(label, 0) -= 1.0f;
    return grad;
}

}  // namespace mnist

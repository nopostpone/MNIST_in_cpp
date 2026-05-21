#include "loss/cross_entropy.h"
#include <cmath>

namespace mnist {

float CrossEntropyLoss::forward(const Eigen::MatrixXf& logits, int label) {
    // Numerically stable softmax + NLL in one pass
    float shift = logits.maxCoeff();
    Eigen::MatrixXf exp_s = (logits.array() - shift).exp();
    float sum_exp = exp_s.sum();
    float inv_sum = 1.0f / sum_exp;

    // Cache probs for backward
    probs_cache_ = exp_s * inv_sum;

    return -std::log(probs_cache_(label, 0) + 1e-12f);
}

Eigen::MatrixXf CrossEntropyLoss::backward(const Eigen::MatrixXf& logits, int label) {
    // dL/d(logits) = softmax(logits) - one_hot(label)
    // Recompute if forward wasn't called with same logits
    float shift = logits.maxCoeff();
    Eigen::MatrixXf exp_s = (logits.array() - shift).exp();
    float inv_sum = 1.0f / exp_s.sum();
    Eigen::MatrixXf grad = exp_s * inv_sum;
    grad(label, 0) -= 1.0f;
    return grad;
}

}  // namespace mnist

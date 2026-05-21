#include "loss/cross_entropy.h"
#include <cmath>

namespace mnist {

float CrossEntropyLoss::forward(const Eigen::MatrixXf& logits, const Eigen::VectorXi& labels) {
    int B = static_cast<int>(logits.cols());
    probs_cache_.resize(logits.rows(), B);

    float total_loss = 0.0f;
    for (int b = 0; b < B; ++b) {
        float shift = logits.col(b).maxCoeff();
        Eigen::MatrixXf exp_s = (logits.col(b).array() - shift).exp();
        float sum_exp = exp_s.sum();
        float inv_sum = 1.0f / sum_exp;
        probs_cache_.col(b) = exp_s * inv_sum;
        total_loss -= std::log(probs_cache_(labels(b), b) + 1e-12f);
    }
    return total_loss;
}

Eigen::MatrixXf CrossEntropyLoss::backward(const Eigen::MatrixXf& logits, const Eigen::VectorXi& labels) {
    int B = static_cast<int>(logits.cols());
    int C = static_cast<int>(logits.rows());
    Eigen::MatrixXf grad(C, B);

    for (int b = 0; b < B; ++b) {
        float shift = logits.col(b).maxCoeff();
        Eigen::MatrixXf exp_s = (logits.col(b).array() - shift).exp();
        float inv_sum = 1.0f / exp_s.sum();
        grad.col(b) = exp_s * inv_sum;
        grad(labels(b), b) -= 1.0f;
    }
    return grad;
}

}  // namespace mnist

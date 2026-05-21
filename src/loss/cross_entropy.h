#pragma once
#include <Eigen/Dense>

namespace mnist {

class CrossEntropyLoss {
    Eigen::MatrixXf probs_cache_;  // softmax output (num_classes, B)
public:
    /// @param logits raw scores, shape (num_classes, B)
    /// @param labels ground truth class ids, shape (B,)
    /// @return      total loss (sum over batch, not averaged)
    float forward(const Eigen::MatrixXf& logits, const Eigen::VectorXi& labels);

    /// Combined gradient: dL/d(logits) = softmax(logits) - one_hot(labels)
    /// Returns (num_classes, B) — each column is gradient for one sample
    Eigen::MatrixXf backward(const Eigen::MatrixXf& logits, const Eigen::VectorXi& labels);

    const Eigen::MatrixXf& probs() const { return probs_cache_; }
};

}  // namespace mnist

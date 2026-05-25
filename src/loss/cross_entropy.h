#pragma once
#include <Eigen/Dense>

namespace mnist {

struct CELoss {
    float loss;
    Eigen::MatrixXf grad;  // (num_classes, B), dL/d(logits)
};

/// Computes softmax + cross-entropy in one atomic call.
/// No mutable state — no ordering dependency between forward and backward.
class CrossEntropyLoss {
public:
    /// @param logits raw scores, shape (num_classes, B)
    /// @param labels ground truth class ids, shape (B,)
    CELoss compute(const Eigen::MatrixXf& logits, const Eigen::VectorXi& labels);
};

}  // namespace mnist

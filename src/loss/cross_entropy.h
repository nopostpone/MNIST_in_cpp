#pragma once
#include <Eigen/Dense>

namespace mnist {

class CrossEntropyLoss {
    Eigen::MatrixXf probs_cache_;  // softmax output, cached for backward
public:
    /// @param logits raw scores from last linear layer, shape (num_classes, 1)
    /// @param label  ground truth class id (0-9)
    /// @return       scalar loss = -log(softmax(logits)[label])
    float forward(const Eigen::MatrixXf& logits, int label);

    /// Combined gradient: dL/d(logits) = softmax(logits) - one_hot(label)
    Eigen::MatrixXf backward(const Eigen::MatrixXf& logits, int label);

    const Eigen::MatrixXf& probs() const { return probs_cache_; }
};

}  // namespace mnist

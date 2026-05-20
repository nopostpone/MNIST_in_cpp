#pragma once
#include <Eigen/Dense>

namespace mnist {

class CrossEntropyLoss {
public:
    /// @param pred  softmax probabilities, shape (num_classes, 1)
    /// @param label ground truth class id (0-9)
    /// @return      scalar loss = -log(pred[label])
    float forward(const Eigen::MatrixXf& pred, int label);

    /// Combined gradient: dL/d(logits) = pred - one_hot(label)
    /// Call this AFTER softmax — it gives gradient w.r.t. logits directly.
    Eigen::MatrixXf backward(const Eigen::MatrixXf& pred, int label);
};

}  // namespace mnist

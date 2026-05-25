#include "loss/cross_entropy.h"
#include <cmath>

namespace mnist {

CELoss CrossEntropyLoss::compute(const Eigen::MatrixXf& logits,
                                  const Eigen::VectorXi& labels) {
    int B = static_cast<int>(logits.cols());
    CELoss result;
    result.grad.resize(logits.rows(), B);
    result.loss = 0.0f;

    for (int b = 0; b < B; ++b) {
        float shift = logits.col(b).maxCoeff();
        Eigen::MatrixXf exp_s = (logits.col(b).array() - shift).exp();
        float inv_sum = 1.0f / exp_s.sum();
        Eigen::MatrixXf probs = exp_s * inv_sum;
        result.loss -= std::log(probs(labels(b), 0) + 1e-12f);
        result.grad.col(b) = probs.col(0);
        result.grad(labels(b), b) -= 1.0f;
    }
    return result;
}

}  // namespace mnist

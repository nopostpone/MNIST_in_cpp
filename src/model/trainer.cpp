#include "model/trainer.h"
#include <iostream>
#include <iomanip>

namespace mnist {

Trainer::Trainer(Sequential& model, CrossEntropyLoss& criterion, float lr)
    : model_(model), criterion_(criterion), lr_(lr)
{
    params_ = model_.parameters();
}

void Trainer::step() {
    for (auto* p : params_) {
        p->data.noalias() -= lr_ * p->grad;
        p->grad.setZero();
    }
}

EpochResult Trainer::train_epoch(const Dataset& train_data,
                                  const Dataset& test_data,
                                  int epoch, int batch_size) {
    float total_loss = 0;
    int   correct    = 0;
    int   n          = static_cast<int>(train_data.images.rows());

    for (int i = 0; i < n; ++i) {
        // ── Sample → (1, 784) matrix ────────────────────────────────────
        Eigen::MatrixXf x(1, 784);
        x = train_data.images.row(i);

        // ── Forward: model outputs logits (10, 1) ───────────────────────
        auto logits = model_.forward(x);

        int label = train_data.labels(i);
        float loss = criterion_.forward(logits, label);
        total_loss += loss;

        // accuracy
        Eigen::Index pred;
        logits.col(0).maxCoeff(&pred);  // argmax works on logits
        if (static_cast<int>(pred) == label) ++correct;

        // ── Backward: loss → dL/d(logits) → propagate ──────────────────
        auto dlogits = criterion_.backward(logits, label);
        model_.backward(dlogits);

        // ── Update ──────────────────────────────────────────────────────
        if ((i + 1) % batch_size == 0 || i == n - 1)
            step();
    }

    EpochResult res;
    res.epoch      = epoch;
    res.train_loss = total_loss / n;
    res.train_acc  = static_cast<float>(correct) / n;
    res.test_acc   = evaluate(test_data);
    return res;
}

float Trainer::evaluate(const Dataset& data) {
    int correct = 0;
    int n = static_cast<int>(data.images.rows());

    for (int i = 0; i < n; ++i) {
        Eigen::MatrixXf x(1, 784);
        x = data.images.row(i);

        auto logits = model_.forward(x);

        Eigen::Index pred;
        logits.col(0).maxCoeff(&pred);
        if (static_cast<int>(pred) == data.labels(i)) ++correct;
    }
    return static_cast<float>(correct) / n;
}

}  // namespace mnist

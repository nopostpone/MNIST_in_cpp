#include "model/trainer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

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

    for (int i = 0; i < n; i += batch_size) {
        int B = std::min(batch_size, n - i);

        // Build batch input (784, B) and label vector
        Eigen::MatrixXf batch_input(784, B);
        Eigen::VectorXi batch_labels(B);
        for (int j = 0; j < B; ++j) {
            batch_input.col(j) = train_data.images.row(i + j).transpose();
            batch_labels(j)    = train_data.labels(i + j);
        }

        // Forward
        auto logits = model_.forward(batch_input);  // (10, B)

        // Loss (total over batch, not averaged)
        float loss = criterion_.forward(logits, batch_labels);
        total_loss += loss;

        // Accuracy
        for (int j = 0; j < B; ++j) {
            Eigen::Index pred;
            logits.col(j).maxCoeff(&pred);
            if (static_cast<int>(pred) == batch_labels(j)) ++correct;
        }

        // Backward
        auto dlogits = criterion_.backward(logits, batch_labels);
        model_.backward(dlogits);

        // Update
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
        Eigen::MatrixXf x(784, 1);
        x.col(0) = data.images.row(i).transpose();

        auto logits = model_.forward(x);

        Eigen::Index pred;
        logits.col(0).maxCoeff(&pred);
        if (static_cast<int>(pred) == data.labels(i)) ++correct;
    }
    return static_cast<float>(correct) / n;
}

}  // namespace mnist

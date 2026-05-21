#pragma once
#include "model/model.h"
#include "loss/cross_entropy.h"
#include "data/mnist_loader.h"
#include <vector>

namespace mnist {

struct EpochResult {
    int   epoch;
    float train_loss;
    float train_acc;
    float test_acc;
};

class Trainer {
    Sequential&        model_;
    CrossEntropyLoss&  criterion_;
    float              lr_;

    std::vector<Parameter*> params_;  // cached for update

public:
    Trainer(Sequential& model, CrossEntropyLoss& criterion, float lr);

    /// Run one epoch, return (avg_loss, train_accuracy).
    EpochResult train_epoch(const Dataset& train_data,
                            const Dataset& test_data,
                            int epoch, int batch_size = 32);

    /// Evaluate accuracy on a dataset (no grad).
    float evaluate(const Dataset& data);

    void set_lr(float lr) { lr_ = lr; }

private:
    void step();  // update + zero_grad for one sample (online SGD)
};

}  // namespace mnist

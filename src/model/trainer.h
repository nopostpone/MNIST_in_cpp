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
    float              momentum_ = 0.0f;
    float              decay_    = 1.0f;
    bool   augment_       = false;
    float  augment_angle_ = 0.0f;
    float  augment_scale_ = 0.0f;
    float  augment_shift_ = 0.0f;

    std::vector<Parameter*> params_;           // cached for update
    std::vector<Eigen::MatrixXf> velocity_;    // for momentum

public:
    Trainer(Sequential& model, CrossEntropyLoss& criterion, float lr);

    /// Run one epoch, return (avg_loss, train_accuracy).
    EpochResult train_epoch(const Dataset& train_data,
                            const Dataset& test_data,
                            int epoch, int batch_size = 32);

    /// Evaluate accuracy on a dataset (no grad).
    float evaluate(const Dataset& data);

    void set_lr(float lr)         { lr_ = lr; }
    void set_momentum(float m)    { momentum_ = m; }
    void set_decay(float d)       { decay_ = d; }
    void set_augment(float angle, float scale, float shift) {
        augment_ = true; augment_angle_ = angle; augment_scale_ = scale; augment_shift_ = shift;
    }

private:
    void step();  // update + zero_grad
    void augment(float* img);  // in-place data augmentation
};

}  // namespace mnist

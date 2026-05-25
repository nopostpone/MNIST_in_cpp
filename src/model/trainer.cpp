#include "model/trainer.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace mnist {

Trainer::Trainer(Sequential& model, CrossEntropyLoss& criterion, float lr)
    : model_(model), criterion_(criterion), lr_(lr)
{
    params_ = model_.parameters();
}

void Trainer::step(float lr) {
    for (size_t i = 0; i < params_.size(); ++i) {
        auto* p = params_[i];
        if (momentum_ > 0.0f) {
            if (velocity_.empty()) velocity_.resize(params_.size());
            if (velocity_[i].size() == 0)
                velocity_[i] = Eigen::MatrixXf::Zero(p->grad.rows(), p->grad.cols());
            velocity_[i] = momentum_ * velocity_[i] + lr * p->grad;
            p->data.noalias() -= velocity_[i];
        } else {
            p->data.noalias() -= lr * p->grad;
        }
        p->grad.setZero();
    }
}

// Augment: random rotation + scale + shift, bilinear interpolation
void Trainer::augment(float* img) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float angle = augment_angle_ * dist(rng) * 3.14159265f / 180.0f;
    float scale = 1.0f + augment_scale_ * dist(rng);
    float dx    = augment_shift_ * dist(rng);
    float dy    = augment_shift_ * dist(rng);

    float c = std::cos(angle), s = std::sin(angle);
    float sc_inv = 1.0f / scale;

    float tmp[784];
    std::copy(img, img + 784, tmp);

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            // Map output (x,y) back to source via inverse transform
            float sx = (x - 13.5f) * sc_inv - dx;
            float sy = (y - 13.5f) * sc_inv + dy;
            float src_x =  c * sx + s * sy + 13.5f;
            float src_y = -s * sx + c * sy + 13.5f;

            int ix0 = static_cast<int>(std::floor(src_x));
            int iy0 = static_cast<int>(std::floor(src_y));
            int ix1 = ix0 + 1, iy1 = iy0 + 1;
            float fx = src_x - ix0, fy = src_y - iy0;

            auto get = [&](int px, int py) -> float {
                if (px < 0 || px >= 28 || py < 0 || py >= 28) return 0.0f;
                return tmp[py * 28 + px];
            };

            float v00 = get(ix0, iy0), v10 = get(ix1, iy0);
            float v01 = get(ix0, iy1), v11 = get(ix1, iy1);

            float val = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                      + v01 * (1 - fx) * fy + v11 * fx * fy;

            img[y * 28 + x] = std::max(0.0f, std::min(1.0f, val));
        }
    }
}

EpochResult Trainer::train_epoch(const Dataset& train_data,
                                  const Dataset& test_data,
                                  int epoch, int batch_size) {
    model_.set_training(true);
    float total_loss = 0;
    int   correct    = 0;
    int   n          = static_cast<int>(train_data.images.rows());
    int   in_dim     = static_cast<int>(train_data.images.cols());

    float epoch_lr = lr_ * std::pow(decay_, epoch - 1);

    for (int i = 0; i < n; i += batch_size) {
        int B = std::min(batch_size, n - i);

        Eigen::MatrixXf batch_input(in_dim, B);
        Eigen::VectorXi batch_labels(B);
        for (int j = 0; j < B; ++j) {
            Eigen::RowVectorXf row = train_data.images.row(i + j);
            float tmp[784];
            for (int k = 0; k < in_dim; ++k) tmp[k] = row(k);
            if (augment_) augment(tmp);
            for (int k = 0; k < in_dim; ++k) batch_input(k, j) = tmp[k];
            batch_labels(j) = train_data.labels(i + j);
        }

        auto logits = model_.forward(batch_input);

        auto cl = criterion_.compute(logits, batch_labels);
        total_loss += cl.loss;

        for (int j = 0; j < B; ++j) {
            Eigen::Index pred;
            logits.col(j).maxCoeff(&pred);
            if (static_cast<int>(pred) == batch_labels(j)) ++correct;
        }

        model_.backward(cl.grad);
        step(epoch_lr);
    }

    EpochResult res;
    res.epoch      = epoch;
    res.train_loss = total_loss / n;
    res.train_acc  = static_cast<float>(correct) / n;
    res.test_acc   = evaluate(test_data);
    return res;
}

float Trainer::evaluate(const Dataset& data) {
    model_.set_training(false);
    int correct = 0;
    int n = static_cast<int>(data.images.rows());
    int in_dim = static_cast<int>(data.images.cols());
    const int batch_size = 64;

    for (int i = 0; i < n; i += batch_size) {
        int B = std::min(batch_size, n - i);
        Eigen::MatrixXf batch_input(in_dim, B);
        for (int j = 0; j < B; ++j)
            batch_input.col(j) = data.images.row(i + j).transpose();

        auto logits = model_.forward(batch_input);

        for (int j = 0; j < B; ++j) {
            Eigen::Index pred;
            logits.col(j).maxCoeff(&pred);
            if (static_cast<int>(pred) == data.labels(i + j)) ++correct;
        }
    }
    return static_cast<float>(correct) / n;
}

}  // namespace mnist

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include "data/mnist_loader.h"
#include "model/model.h"
#include "model/trainer.h"
#include "layers/conv2d.h"
#include "layers/max_pool.h"
#include "layers/flatten.h"
#include "layers/fully_connected.h"
#include "activations/relu.h"
#include "loss/cross_entropy.h"
#include "core/weight_io.h"

static mnist::Sequential build_model() {
    mnist::Sequential model;
    model.add<mnist::Conv2D>(1, 32, 3, 1, 1, 28, 28);
    model.add<mnist::ReLU>();
    model.add<mnist::MaxPool2D>(32, 28, 28, 2, 2);
    model.add<mnist::Conv2D>(32, 64, 3, 1, 0, 14, 14);
    model.add<mnist::ReLU>();
    model.add<mnist::MaxPool2D>(64, 12, 12, 2, 2);
    model.add<mnist::Flatten>(64, 6, 6);
    model.add<mnist::Linear>(2304, 128);
    model.add<mnist::ReLU>();
    model.add<mnist::Linear>(128, 10);
    return model;
}

void do_train() {
    std::cout << "Loading MNIST..." << std::endl;
    auto train = mnist::Loader::load_training("data", true);
    auto test  = mnist::Loader::load_test("data", true);
    std::cout << "Train: " << train.images.rows()
              << "  Test: " << test.images.rows() << std::endl;

    auto model = build_model();
    mnist::CrossEntropyLoss criterion;
    mnist::Trainer trainer(model, criterion, 0.001f);

    std::cout << std::fixed << std::setprecision(4);
    namespace chr = std::chrono;
    auto t0 = chr::steady_clock::now();

    for (int epoch = 1; epoch <= 5; ++epoch) {
        auto r = trainer.train_epoch(train, test, epoch, 64);
        auto t1 = chr::steady_clock::now();
        double sec = chr::duration<double>(t1 - t0).count();
        std::cout << "Epoch " << r.epoch
                  << " | loss: " << r.train_loss
                  << " | train_acc: " << r.train_acc
                  << " | test_acc: " << r.test_acc
                  << " | " << int(sec) << "s" << std::endl;
        t0 = t1;
    }

    // Save trained weights
    auto params = model.parameters();
    mnist::save_weights(params, "model_weights.bin");
    std::cout << "Weights saved to model_weights.bin" << std::endl;
}

void do_predict() {
    auto model = build_model();
    auto params = model.parameters();
    mnist::load_weights(params, "model_weights.bin");

    // Read 784 raw bytes from stdin (0-255 grayscale)
    unsigned char buf[784];
    std::cin.read(reinterpret_cast<char*>(buf), 784);

    Eigen::MatrixXf x(784, 1);
    const float scale = 1.0f / 255.0f;
    for (int i = 0; i < 784; ++i)
        x(i, 0) = static_cast<float>(buf[i]) * scale;

    auto logits = model.forward(x);

    // Softmax
    float shift = logits.maxCoeff();
    Eigen::MatrixXf exp_s = (logits.array() - shift).exp();
    Eigen::MatrixXf probs = (exp_s / exp_s.sum()).eval();

    Eigen::Index pred;
    probs.col(0).maxCoeff(&pred);
    std::cout << static_cast<int>(pred);
    for (int i = 0; i < 10; ++i)
        std::cout << " " << probs(i, 0);
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "predict") == 0) {
        do_predict();
    } else {
        do_train();
    }
    return 0;
}

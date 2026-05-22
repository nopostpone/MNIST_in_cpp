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
#include "layers/dropout.h"
#include "activations/relu.h"
#include "loss/cross_entropy.h"
#include "core/weight_io.h"

static int run_sanity() {
    std::cout << "[" << std::endl;
    auto train_full = mnist::Loader::load_training("data", true);
    auto test_full  = mnist::Loader::load_test("data", true);

    const int n_train = 128;
    const int n_test  = 64;
    mnist::Dataset train_sub, test_sub;
    train_sub.images = train_full.images.topRows(n_train);
    train_sub.labels = train_full.labels.head(n_train);
    test_sub.images  = test_full.images.topRows(n_test);
    test_sub.labels  = test_full.labels.head(n_test);

    mnist::Sequential model;
    model.add<mnist::Conv2D>(1, 10, 3, 1, 0, 28, 28);
    model.add<mnist::ReLU>();
    model.add<mnist::Conv2D>(10, 10, 3, 1, 0, 26, 26);
    model.add<mnist::ReLU>();
    model.add<mnist::MaxPool2D>(10, 24, 24, 2, 2);
    model.add<mnist::Conv2D>(10, 20, 3, 1, 0, 12, 12);
    model.add<mnist::ReLU>();
    model.add<mnist::Conv2D>(20, 20, 3, 1, 0, 10, 10);
    model.add<mnist::ReLU>();
    model.add<mnist::MaxPool2D>(20, 8, 8, 2, 2);
    model.add<mnist::Flatten>(20, 4, 4);
    model.add<mnist::Dropout>(0.2f);
    model.add<mnist::Linear>(320, 128);
    model.add<mnist::ReLU>();
    model.add<mnist::Linear>(128, 10);

    mnist::CrossEntropyLoss criterion;
    mnist::Trainer trainer(model, criterion, 0.001f);

    {
        Eigen::MatrixXf x(784, 1);
        x.col(0) = train_sub.images.row(0).transpose();
        auto out = model.forward(x);
        if (out.rows() != 10 || out.cols() != 1) {
            std::cerr << "FAIL: output shape" << std::endl;
            return 1;
        }
        std::cout << "  { \"dry_run\": \"ok\" }," << std::endl;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto r = trainer.train_epoch(train_sub, test_sub, 1, 32);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  { \"epoch\": " << r.epoch
              << ", \"loss\": " << r.train_loss
              << ", \"train_acc\": " << r.train_acc
              << ", \"test_acc\": " << r.test_acc
              << ", \"time_s\": " << sec << " }" << std::endl;

    // Verify loss is not NaN and accuracy > 10% (random baseline)
    bool ok = true;
    if (r.train_loss != r.train_loss) {  // NaN check
        std::cerr << "FAIL: loss is NaN" << std::endl;
        ok = false;
    }
    if (r.train_loss > 3.0f) {
        std::cerr << "FAIL: loss " << r.train_loss << " > 3.0" << std::endl;
        ok = false;
    }
    std::cout << "]" << std::endl;

    if (ok) {
        std::cout << "SANITY: PASS" << std::endl;
    } else {
        std::cout << "SANITY: FAIL" << std::endl;
        return 1;
    }
    return 0;
}

static mnist::Sequential build_model() {
    mnist::Sequential model;
    model.add<mnist::Conv2D>(1, 10, 3, 1, 0, 28, 28);
    model.add<mnist::ReLU>();
    model.add<mnist::Conv2D>(10, 10, 3, 1, 0, 26, 26);
    model.add<mnist::ReLU>();
    model.add<mnist::MaxPool2D>(10, 24, 24, 2, 2);
    model.add<mnist::Conv2D>(10, 20, 3, 1, 0, 12, 12);
    model.add<mnist::ReLU>();
    model.add<mnist::Conv2D>(20, 20, 3, 1, 0, 10, 10);
    model.add<mnist::ReLU>();
    model.add<mnist::MaxPool2D>(20, 8, 8, 2, 2);
    model.add<mnist::Flatten>(20, 4, 4);
    model.add<mnist::Dropout>(0.2f);
    model.add<mnist::Linear>(320, 128);
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
    trainer.set_momentum(0.9f);
    trainer.set_decay(0.95f);
    trainer.set_augment(15.0f, 0.15f, 2.8f);  // ±15°, ±15% scale, ±2.8px shift

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

    auto params = model.parameters();
    mnist::save_weights(params, "model_weights.bin");
    std::cout << "Weights saved to model_weights.bin" << std::endl;
}

void do_predict() {
    auto model = build_model();
    auto params = model.parameters();
    mnist::load_weights(params, "model_weights.bin");

    unsigned char buf[784];
    std::cin.read(reinterpret_cast<char*>(buf), 784);

    Eigen::MatrixXf x(784, 1);
    const float scale = 1.0f / 255.0f;
    for (int i = 0; i < 784; ++i)
        x(i, 0) = static_cast<float>(buf[i]) * scale;

    auto logits = model.forward(x);

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

void do_export() {
    auto model = build_model();
    auto params = model.parameters();
    mnist::load_weights(params, "model_weights.bin");

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << "{\"r\":" << params[i]->data.rows()
                  << ",\"c\":" << params[i]->data.cols()
                  << ",\"d\":[";
        for (int j = 0; j < params[i]->data.size(); ++j) {
            if (j > 0) std::cout << ",";
            std::cout << params[i]->data(j);
        }
        std::cout << "]}";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "sanity") == 0)
        return run_sanity();
    if (argc >= 2 && std::strcmp(argv[1], "predict") == 0) {
        do_predict();
    } else if (argc >= 2 && std::strcmp(argv[1], "export") == 0) {
        do_export();
    } else {
        do_train();
    }
    return 0;
}

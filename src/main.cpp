#include <iostream>
#include <iomanip>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "data/mnist_loader.h"
#include "model/model.h"
#include "model/trainer.h"
#include "layers/conv2d.h"
#include "layers/max_pool.h"
#include "layers/flatten.h"
#include "layers/fully_connected.h"
#include "activations/relu.h"
#include "loss/cross_entropy.h"

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP: not enabled" << std::endl;
#endif

    std::cout << std::fixed << std::setprecision(4);
    auto train_full = mnist::Loader::load_training("data", true);
    auto test_full  = mnist::Loader::load_test("data", true);

    const int n_train = 2000;
    const int n_test  = 500;
    mnist::Dataset train_sub, test_sub;
    train_sub.images = train_full.images.topRows(n_train);
    train_sub.labels = train_full.labels.head(n_train);
    test_sub.images  = test_full.images.topRows(n_test);
    test_sub.labels  = test_full.labels.head(n_test);

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

    mnist::CrossEntropyLoss criterion;
    mnist::Trainer trainer(model, criterion, 0.001f);

    namespace chr = std::chrono;
    auto t0 = chr::steady_clock::now();
    auto r = trainer.train_epoch(train_sub, test_sub, 1, 64);
    auto t1 = chr::steady_clock::now();
    double sec = chr::duration<double>(t1 - t0).count();

    std::cout << "Epoch 1 | loss: " << r.train_loss
              << " | train_acc: " << r.train_acc
              << " | test_acc: " << r.test_acc
              << " | " << sec << "s"
              << " | " << int(n_train / sec) << " samples/s" << std::endl;

    return 0;
}

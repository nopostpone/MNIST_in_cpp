#include <iostream>
#include "data/mnist_loader.h"

int main() {
    std::cout << "1. loading..." << std::endl;
    auto train = mnist::Loader::load_training("data", true);
    std::cout << "2. loaded " << train.images.rows() << " images" << std::endl;
    std::cout << "3. done" << std::endl;
    return 0;
}

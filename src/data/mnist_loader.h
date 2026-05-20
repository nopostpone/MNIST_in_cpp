#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cstdint>

namespace mnist {

struct Dataset {
    Eigen::MatrixXf images;  // N x 784, each row is one flattened 28x28 image
    Eigen::VectorXi labels;  // N elements, values 0-9
};

struct Loader {
    /// Load training set (60000 samples) from data/ directory
    static Dataset load_training(const std::string& data_dir, bool normalize = true);

    /// Load test set (10000 samples) from data/ directory
    static Dataset load_test(const std::string& data_dir, bool normalize = true);

private:
    static Dataset load(const std::string& image_path,
                        const std::string& label_path,
                        bool normalize);
};

}  // namespace mnist

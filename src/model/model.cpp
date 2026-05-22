#include "model/model.h"

namespace mnist {

Eigen::MatrixXf Sequential::forward(const Eigen::MatrixXf& x) {
    Eigen::MatrixXf act = x;
    for (auto& layer : layers_)
        act = layer->forward(act);
    return act;
}

void Sequential::backward(const Eigen::MatrixXf& grad) {
    Eigen::MatrixXf g = grad;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        g = (*it)->backward(g);
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> params;
    for (auto& layer : layers_)
        for (auto* p : layer->parameters())
            params.push_back(p);
    return params;
}

void Sequential::set_training(bool t) {
    for (auto& layer : layers_) layer->set_training(t);
}

}  // namespace mnist

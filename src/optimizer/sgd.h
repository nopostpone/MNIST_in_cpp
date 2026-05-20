#pragma once
#include "core/module.h"

namespace mnist {

class SGD {
    float lr_;
    std::vector<Module*> mods_;
public:
    SGD(float lr, std::vector<Module*> modules)
        : lr_(lr), mods_(std::move(modules)) {}

    void step() {
        for (auto* m : mods_) {
            m->update(lr_);
            m->zero_grad();
        }
    }

    void set_lr(float lr) { lr_ = lr; }
    float lr() const { return lr_; }
};

}  // namespace mnist

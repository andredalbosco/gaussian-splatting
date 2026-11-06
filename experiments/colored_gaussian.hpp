#pragma once

#include "gaussian.hpp"

#include <algorithm>
#include <span>

struct ColoredGaussian {
    Eigen::Vector4f position;
    Eigen::Vector4f scale;
    Eigen::Vector4f rotation;
    Eigen::Vector4f color;
};

inline std::vector<Gaussian> toGaussians(std::span<const ColoredGaussian> data) {
    std::vector<Gaussian> gaussians;
    gaussians.reserve(data.size());
    for (const auto& g : data) {
        Gaussian gaussian {g.position.head<3>(), g.color.w(), g.scale, g.rotation, {}};
        for (int channel = 0; channel < 3; channel++) {
            gaussian.sh.coefficients[0][channel] = (g.color[channel] - 0.5f) / 0.2820947918f;
        }
        gaussians.push_back(gaussian);
    }
    return gaussians;
}

inline std::vector<ColoredGaussian> readColoredPlyFile(const char* path) {
    auto gaussians = readPlyFile(path);
    std::vector<ColoredGaussian> colored(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); i++) {
        const auto& g = gaussians[i];
        colored[i] = {{g.position.x(), g.position.y(), g.position.z(), 0}, g.scale, g.rotation,
            {0, 0, 0, g.opacity}};
        for (int channel = 0; channel < 3; channel++) {
            colored[i].color[channel] = std::clamp(0.5f + 0.2820947918f * g.sh.coefficients[0][channel], 0.0f, 1.0f);
        }
    }
    return colored;
}

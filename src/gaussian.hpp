#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

struct SHCoefficients {
    std::array<std::array<float, 4>, 16> coefficients {};
};

struct Gaussian {
    Eigen::Vector3f position;
    float opacity;
    Eigen::Vector4f scale;
    Eigen::Vector4f rotation;
    SHCoefficients sh;
};

static_assert(sizeof(Gaussian) == 304);

std::vector<Gaussian> readPlyFile(const char* path);

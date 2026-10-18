#pragma once

#include <vector>

#include <Eigen/Core>

struct Gaussian {
    Eigen::Vector4f position;
    Eigen::Vector4f scale;
    Eigen::Vector4f rotation;
    Eigen::Vector4f color;
};

std::vector<Gaussian> readPlyFile(const char* path);

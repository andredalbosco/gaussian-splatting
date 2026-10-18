#include "gaussian.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include <Eigen/Geometry>

std::vector<Gaussian> readPlyFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    size_t vertexCount = 0;
    std::vector<std::string> properties;

    for (std::string line; std::getline(file, line) && line != "end_header";) {
        std::istringstream header(line);
        std::string kind;
        std::string type;
        std::string name;
        header >> kind;

        if (kind == "element") {
            header >> name;
            if (name == "vertex") header >> vertexCount;
        }

        if (kind == "property") {
            header >> type >> name;
            properties.push_back(name);
        }
    }

    auto offset = [&](const char* name) {
        return std::find(properties.begin(), properties.end(), name) - properties.begin();
    };

    size_t positionOffset = offset("x");
    size_t colorOffset = offset("f_dc_0");
    size_t scaleOffset = offset("scale_0");
    size_t rotationOffset = offset("rot_0");
    size_t opacityOffset = offset("opacity");

    std::vector<float> row(properties.size());
    std::vector<Gaussian> gaussians(vertexCount);

    for (Gaussian& gaussian : gaussians) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));

        gaussian.position.head<3>() =
            Eigen::Map<const Eigen::Vector3f>(row.data() + positionOffset);
        gaussian.position.w() = 0;

        gaussian.scale.head<3>() =
            Eigen::Map<const Eigen::Vector3f>(row.data() + scaleOffset).array().exp().matrix();
        gaussian.scale.w() = 0;

        Eigen::Quaternionf rotation {
            row[rotationOffset], row[rotationOffset + 1],
            row[rotationOffset + 2], row[rotationOffset + 3],
        };
        gaussian.rotation = rotation.normalized().coeffs();

        Eigen::Vector3f color = Eigen::Map<const Eigen::Vector3f>(row.data() + colorOffset);
        gaussian.color.head<3>() =
            (0.5f + 0.2820947918f * color.array()).max(0.0f).min(1.0f).matrix();
        gaussian.color.w() = 1.0f / (1.0f + std::exp(-row[opacityOffset]));
    }

    return gaussians;
}

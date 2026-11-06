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

    size_t restCount = std::count_if(properties.begin(), properties.end(), [](const std::string& name) {
        return name.starts_with("f_rest_");
    }) / 3;
    std::vector<size_t> restOffsets(3 * restCount);
    for (size_t i = 0; i < restOffsets.size(); i++) {
        restOffsets[i] = offset(("f_rest_" + std::to_string(i)).c_str());
    }

    std::vector<float> row(properties.size());
    std::vector<Gaussian> gaussians(vertexCount);

    for (size_t i = 0; i < vertexCount; i++) {
        Gaussian& gaussian = gaussians[i];
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));

        gaussian.position.head<3>() =
            Eigen::Map<const Eigen::Vector3f>(row.data() + positionOffset);

        gaussian.scale.head<3>() =
            Eigen::Map<const Eigen::Vector3f>(row.data() + scaleOffset).array().exp().matrix();
        gaussian.scale.w() = 0;

        Eigen::Quaternionf rotation {
            row[rotationOffset], row[rotationOffset + 1],
            row[rotationOffset + 2], row[rotationOffset + 3],
        };
        gaussian.rotation = rotation.normalized().coeffs();

        for (size_t channel = 0; channel < 3; channel++) {
            gaussian.sh.coefficients[0][channel] = row[colorOffset + channel];
            for (size_t coefficient = 0; coefficient < std::min(restCount, size_t(15)); coefficient++) {
                gaussian.sh.coefficients[coefficient + 1][channel] =
                    row[restOffsets[channel * restCount + coefficient]];
            }
        }
        gaussian.opacity = 1.0f / (1.0f + std::exp(-row[opacityOffset]));
    }

    return gaussians;
}

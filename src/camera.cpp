#include "camera.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Geometry>

Eigen::Matrix4f viewMatrix(
    const Eigen::Vector3f& eye,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& up
) {
    Eigen::Vector3f forward = (center - eye).normalized();
    Eigen::Vector3f right = forward.cross(up).normalized();
    Eigen::Vector3f cameraUp = right.cross(forward);

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view.block<1, 3>(0, 0) = right.transpose();
    view.block<1, 3>(1, 0) = cameraUp.transpose();
    view.block<1, 3>(2, 0) = forward.transpose();
    view(0, 3) = -right.dot(eye);
    view(1, 3) = -cameraUp.dot(eye);
    view(2, 3) = -forward.dot(eye);

    return view;
}

Eigen::Matrix4f projectionMatrix(float fovDegrees, float aspect, float near, float far) {
    float focal = 1.0f / std::tan(0.5f * fovDegrees * std::numbers::pi_v<float> / 180.0f);

    Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
    projection(0, 0) = focal / aspect;
    projection(1, 1) = focal;
    projection(2, 2) = far / (far - near);
    projection(2, 3) = -near * far / (far - near);
    projection(3, 2) = 1.0f;

    return projection;
}

Eigen::Vector3f OrbitCamera::position() const {
    float yawRadians = yaw * std::numbers::pi_v<float> / 180.0f;
    float pitchRadians = pitch * std::numbers::pi_v<float> / 180.0f;
    return center + distance * Eigen::Vector3f {
        std::sin(yawRadians) * std::cos(pitchRadians),
        std::sin(pitchRadians),
        std::cos(yawRadians) * std::cos(pitchRadians),
    };
}

Eigen::Matrix4f OrbitCamera::viewProjection(float aspect) const {
    Eigen::Matrix4f view = viewMatrix(position(), center, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f projection = projectionMatrix(fovy, aspect, zNear, zFar);

    return projection * view;
}

void OrbitCamera::orbit(const Eigen::Vector2f& delta) {
    yaw -= delta.x() * orbitSensitivity;
    pitch = std::clamp(pitch + delta.y() * orbitSensitivity, -maxPitch, maxPitch);
}

void OrbitCamera::zoom(float delta) {
    distance = std::clamp(
        distance * std::exp(-zoomSensitivity * delta),
        minDistance,
        maxDistance
    );
}

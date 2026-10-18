#pragma once

#include <Eigen/Core>

struct OrbitCamera {
    float fovy = 60;
    float zNear = 0.01f;
    float zFar = 100.0f;
    float orbitSensitivity = 0.3f;
    float zoomSensitivity = 0.1f;
    float maxPitch = 85;
    float minDistance = 0.3f;
    float maxDistance = 20.0f;

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    float distance = 3.5f;
    float yaw = 10;
    float pitch = 5;

    Eigen::Vector3f position() const;
    Eigen::Matrix4f viewProjection(float aspect) const;

    void orbit(const Eigen::Vector2f& delta);
    void zoom(float delta);
};

Eigen::Matrix4f viewMatrix(
    const Eigen::Vector3f& eye,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& up
);
Eigen::Matrix4f projectionMatrix(float fovDegrees, float aspect, float near, float far);

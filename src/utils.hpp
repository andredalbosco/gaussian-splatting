#pragma once

#include "gaussian.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

struct GLFWwindow;

struct Extent2D {
    uint32_t width;
    uint32_t height;
};

Extent2D getFramebufferSize(GLFWwindow* window);
bool checkFramebufferSize(Extent2D size);

wgpu::Instance createInstance();
wgpu::Surface createSurface(const wgpu::Instance& instance, GLFWwindow* window);
wgpu::Adapter requestAdapter(const wgpu::Instance& instance, const wgpu::Surface& surface);
wgpu::Device requestDevice(const wgpu::Instance& instance, const wgpu::Adapter& adapter);
wgpu::SurfaceConfiguration configureSurface(
    const wgpu::Device& device,
    const wgpu::Adapter& adapter,
    const wgpu::Surface& surface,
    GLFWwindow* window
);

void initGlfw(int width, int height, const char* title, GLFWwindow*& window);
void shutdownGlfw(GLFWwindow* window);

void initWgpu(
    GLFWwindow* window,
    wgpu::Instance& instance,
    wgpu::Device& device,
    wgpu::Surface& surface,
    wgpu::SurfaceConfiguration& surfaceConfig
);
void shutdownWgpu(wgpu::Surface& surface);

bool handleWindowResize(
    const wgpu::Surface& surface,
    GLFWwindow* window,
    wgpu::SurfaceConfiguration& surfaceConfig
);

std::string readFile(const char* path);
std::vector<Gaussian> readPlyFile(const char* path);
wgpu::ShaderModule readWgslFile(const wgpu::Device& device, const char* path);

#pragma once

#include <webgpu/webgpu_cpp.h>

struct GLFWwindow;

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

wgpu::ShaderModule readWgslFile(const wgpu::Device& device, const char* path);

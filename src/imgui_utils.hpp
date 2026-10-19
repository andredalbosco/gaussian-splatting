#pragma once

#include <webgpu/webgpu_cpp.h>

struct GLFWwindow;

void initImgui(
    const wgpu::Device& device,
    GLFWwindow* window,
    wgpu::TextureFormat colorFormat,
    wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Undefined
);
void shutdownImgui();

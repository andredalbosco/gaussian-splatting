#pragma once

#include "glfw_utils.hpp"
#include "wgpu_utils.hpp"

struct BenchmarkResult {
    double gpuMedianMs;
    double gpuP90Ms;
    double throughputMs;
    std::vector<uint8_t> pixels;
};

BenchmarkResult measureRendering(
    const wgpu::Instance& instance,
    const wgpu::Device& device,
    const wgpu::RenderPipeline& pipeline,
    const wgpu::BindGroup& bindGroup,
    wgpu::TextureFormat colorFormat,
    Extent2D size,
    uint32_t gaussianCount
);

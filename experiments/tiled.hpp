#pragma once

#include "colored_gaussian.hpp"

#include <array>
#include <span>
#include <webgpu/webgpu_cpp.h>

struct TiledRenderer {
    struct Stats { uint32_t visible, pairs, overflow, padding; } stats {};
    wgpu::Device device;
    wgpu::ComputePipeline project, prepareDispatch, radixLocal, radixPrefix, radixScatter, tileRanges, rasterize;
    wgpu::RenderPipeline presentPipeline;
    std::array<wgpu::BindGroup,7> groups;
    std::array<std::array<wgpu::BindGroup,24>,2> radixGroups;
    wgpu::BindGroup dispatchGroup, presentGroup;
    wgpu::Buffer gaussians, projected, pairs, scratch, histogram, counters, ranges, dispatch, camera, radix;
    wgpu::Buffer readback;
    wgpu::Texture image;
    wgpu::QuerySet timestampQueries;
    uint32_t count = 0;
    uint32_t capacity = 1 << 20;
    uint32_t maxCapacity = 0;
    uint32_t width = 0, height = 0;
    bool statsReady = false, statsPending = false;

    void init(const wgpu::Device& device, wgpu::TextureFormat format);
    void load(std::span<const ColoredGaussian> data);
    void resize(uint32_t width, uint32_t height);
    void reservePairs(uint32_t capacity);
    void createBindGroup();
    void record(const wgpu::CommandEncoder& encoder, const Eigen::Matrix4f& viewProjection,
                uint32_t width, uint32_t height, wgpu::Color background = {0.05, 0.05, 0.07, 1});
    void present(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target);
};

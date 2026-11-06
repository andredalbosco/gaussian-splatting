#include "tiled.hpp"
#include "colored_gaussian.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
struct PackedGaussian {
    float position[3], opacity;
    float rotation[4];
    float scale[3];
    uint32_t color;
};
static_assert(sizeof(PackedGaussian) == 48);

struct CameraData {
    Eigen::Matrix4f viewProjection;
    uint32_t width, height, count, capacity;
    float background[4];
};
static_assert(sizeof(CameraData) == 96);
}

void TiledRenderer::init(const wgpu::Device& gpu, wgpu::TextureFormat format) {
    device = gpu;
    wgpu::Limits limits;
    device.GetLimits(&limits);
    maxCapacity = std::min(uint32_t(limits.maxStorageBufferBindingSize / 16), 65535u * 256u);
    wgpu::BufferDescriptor readbackDesc {.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead, .size = 16};
    readback = device.CreateBuffer(&readbackDesc);
    auto shader = readWgslFile(device, SRC_DIR "/../experiments/tiled.wgsl");
    auto pipeline = [&](const char* entry) {
        wgpu::ComputePipelineDescriptor desc {.label = entry,
            .compute = {.module = shader, .entryPoint = entry}};
        return device.CreateComputePipeline(&desc);
    };
    project = pipeline("project");
    prepareDispatch = pipeline("prepare_dispatch");
    radixLocal = pipeline("radix_local");
    radixPrefix = pipeline("radix_prefix");
    radixScatter = pipeline("radix_scatter");
    tileRanges = pipeline("tile_ranges");
    rasterize = pipeline("rasterize");

    auto presentShader = readWgslFile(device, SRC_DIR "/../experiments/tiled_present.wgsl");
    wgpu::ColorTargetState target {.format = format};
    wgpu::FragmentState fragment {.module = presentShader, .entryPoint = "fs_main", .targetCount = 1, .targets = &target};
    wgpu::RenderPipelineDescriptor desc {.vertex = {.module = presentShader, .entryPoint = "vs_main"}, .fragment = &fragment};
    presentPipeline = device.CreateRenderPipeline(&desc);
}

void TiledRenderer::load(std::span<const ColoredGaussian> data) {
    count = data.size();
    std::vector<PackedGaussian> packed(count);
    for (uint32_t i = 0; i < count; i++) {
        const auto& g = data[i];
        std::copy_n(g.position.data(), 3, packed[i].position);
        std::copy_n(g.rotation.data(), 4, packed[i].rotation);
        std::copy_n(g.scale.data(), 3, packed[i].scale);
        packed[i].opacity = g.color.w();
        packed[i].color = 0xff000000u;
        for (uint32_t c = 0; c < 3; c++)
            packed[i].color |= uint32_t(std::round(std::clamp(g.color[c], 0.0f, 1.0f) * 255)) << (c * 8);
    }
    auto buffer = [&](uint64_t bytes, wgpu::BufferUsage extra = wgpu::BufferUsage::None) {
        wgpu::BufferDescriptor desc {.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | extra,
                                     .size = std::max(uint64_t(16), bytes)};
        return device.CreateBuffer(&desc);
    };
    gaussians = buffer(std::max(uint64_t(48), uint64_t(count) * 48));
    device.GetQueue().WriteBuffer(gaussians, 0, packed.data(), packed.size() * sizeof(PackedGaussian));
    projected = buffer(std::max(uint64_t(32), uint64_t(count) * 32));
    reservePairs(capacity);
    counters = buffer(16, wgpu::BufferUsage::CopySrc);
    dispatch = buffer(12, wgpu::BufferUsage::Indirect);
    wgpu::BufferDescriptor uniform {.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, .size = 96};
    camera = device.CreateBuffer(&uniform);
    uniform.size = 24 * 256;
    radix = device.CreateBuffer(&uniform);
    width = height = 0;
    stats = {};
}

void TiledRenderer::reservePairs(uint32_t newCapacity) {
    capacity = std::min(maxCapacity, newCapacity);
    wgpu::BufferDescriptor desc {.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                                 .size = uint64_t(capacity) * 16};
    pairs = device.CreateBuffer(&desc);
    scratch = device.CreateBuffer(&desc);
    desc.size = uint64_t((capacity + 255) / 256) * 16 * 4;
    histogram = device.CreateBuffer(&desc);
}

void TiledRenderer::resize(uint32_t w, uint32_t h) {
    width = w;
    height = h;
    wgpu::TextureDescriptor desc {.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding |
        wgpu::TextureUsage::CopySrc, .size = {w, h, 1}, .format = wgpu::TextureFormat::RGBA8Unorm};
    image = device.CreateTexture(&desc);
    wgpu::BufferDescriptor rangeDesc {.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = uint64_t((w + 15) / 16) * ((h + 15) / 16) * 8};
    ranges = device.CreateBuffer(&rangeDesc);
    createBindGroup();
}

void TiledRenderer::createBindGroup() {
    wgpu::Buffer buffers[] {camera, gaussians, projected, pairs, scratch, histogram, counters, ranges};
    wgpu::ComputePipeline pipelines[] {project, prepareDispatch, radixLocal, radixPrefix, radixScatter, tileRanges, rasterize};
    const std::vector<uint32_t> bindings[] {{0,1,2,3,6}, {0,6}, {0,3,4,5,6},
        {0,5,6}, {0,3,4,5,6}, {0,3,6,7}, {0,2,3,6,7}};
    auto view = image.CreateView();
    for (uint32_t stage=0; stage<7; stage++) {
        std::vector<wgpu::BindGroupEntry> entries;
        for (auto binding : bindings[stage])
            entries.push_back({.binding = binding, .buffer = buffers[binding], .size = buffers[binding].GetSize()});
        if (stage==6) entries.push_back({.binding = 9, .textureView = view});
        wgpu::BindGroupDescriptor desc {.layout = pipelines[stage].GetBindGroupLayout(0),
            .entryCount = uint32_t(entries.size()), .entries = entries.data()};
        groups[stage] = device.CreateBindGroup(&desc);
    }
    wgpu::BindGroupEntry dispatchEntry {.binding = 1, .buffer = dispatch, .size = dispatch.GetSize()};
    wgpu::BindGroupDescriptor desc {.layout = prepareDispatch.GetBindGroupLayout(1), .entryCount = 1, .entries = &dispatchEntry};
    dispatchGroup = device.CreateBindGroup(&desc);
    for (uint32_t stage=0;stage<2;stage++) for (uint32_t i=0;i<24;i++) {
        wgpu::BindGroupEntry entry {.binding = 0, .buffer = radix, .offset = i*256u, .size = 16};
        desc = {.layout = (stage==0 ? radixLocal : radixScatter).GetBindGroupLayout(1), .entryCount = 1, .entries = &entry};
        radixGroups[stage][i] = device.CreateBindGroup(&desc);
    }
    wgpu::BindGroupEntry presentEntry {.binding = 0, .textureView = view};
    desc = {.layout = presentPipeline.GetBindGroupLayout(0), .entryCount = 1, .entries = &presentEntry};
    presentGroup = device.CreateBindGroup(&desc);
}

void TiledRenderer::record(const wgpu::CommandEncoder& encoder, const Eigen::Matrix4f& viewProjection,
                           uint32_t w, uint32_t h, wgpu::Color background) {
    if (width != w || height != h) resize(w, h);
    if (statsReady && !statsPending) {
        statsPending = true;
        readback.MapAsync(wgpu::MapMode::Read, 0, 16, wgpu::CallbackMode::AllowProcessEvents,
            [this](wgpu::MapAsyncStatus status, wgpu::StringView) {
                if (status == wgpu::MapAsyncStatus::Success) {
                    std::memcpy(&stats, readback.GetConstMappedRange(), sizeof(stats));
                    readback.Unmap();
                }
                statsReady = statsPending = false;
            });
    }
    if (stats.overflow && stats.pairs > capacity && capacity < maxCapacity) {
        reservePairs(std::bit_ceil(stats.pairs));
        createBindGroup();
    }
    CameraData params {viewProjection, w, h, count, capacity,
        {float(background.r), float(background.g), float(background.b), float(background.a)}};
    auto queue = device.GetQueue();
    queue.WriteBuffer(camera, 0, &params, sizeof(params));
    std::array<uint32_t, 24 * 64> digits {};
    uint32_t passes = 0;
    // Stable tie order matches the back-to-front quad renderer (reverse source ID).
    for (auto [component, bits] : {std::pair {3u, std::bit_width(count > 0 ? count - 1 : 0)},
                                   std::pair {0u, 32},
                                   std::pair {1u, std::bit_width(((w + 15) / 16) * ((h + 15) / 16) - 1)}}) {
        for (uint32_t shift = 0; shift < bits; shift += 4) {
            digits[passes * 64] = shift;
            digits[passes * 64 + 1] = component;
            passes++;
        }
    }
    queue.WriteBuffer(radix, 0, digits.data(), sizeof(digits));
    encoder.ClearBuffer(counters);
    encoder.ClearBuffer(ranges);
    wgpu::PassTimestampWrites begin {.querySet = timestampQueries, .beginningOfPassWriteIndex = 0};
    wgpu::PassTimestampWrites end {.querySet = timestampQueries, .endOfPassWriteIndex = 1};
    wgpu::ComputePassDescriptor passDesc {.timestampWrites = timestampQueries ? &begin : nullptr};
    auto pass = encoder.BeginComputePass(&passDesc);
    wgpu::ComputePipeline pipelines[] {project, prepareDispatch, radixLocal, radixPrefix, radixScatter, tileRanges, rasterize};
    auto stage = [&](uint32_t i) {
        pass.SetPipeline(pipelines[i]);
        pass.SetBindGroup(0, groups[i]);
    };
    stage(0);
    pass.DispatchWorkgroups((count + 255) / 256);
    stage(1);
    pass.SetBindGroup(1, dispatchGroup);
    pass.DispatchWorkgroups(1);
    pass.End();
    passDesc.timestampWrites = timestampQueries ? &end : nullptr;
    pass = encoder.BeginComputePass(&passDesc);
    for (uint32_t i = 0; i < passes; i++) {
        stage(2);
        pass.SetBindGroup(1, radixGroups[0][i]);
        pass.DispatchWorkgroupsIndirect(dispatch, 0);
        stage(3);
        pass.DispatchWorkgroups(1);
        stage(4);
        pass.SetBindGroup(1, radixGroups[1][i]);
        pass.DispatchWorkgroupsIndirect(dispatch, 0);
    }
    stage(5);
    pass.DispatchWorkgroupsIndirect(dispatch, 0);
    stage(6);
    pass.DispatchWorkgroups((w + 15) / 16, (h + 15) / 16);
    pass.End();
    if (!statsReady && !statsPending) {
        encoder.CopyBufferToBuffer(counters, 0, readback, 0, 16);
        statsReady = true;
    }
}

void TiledRenderer::present(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target) {
    wgpu::RenderPassColorAttachment color {.view = target, .loadOp = wgpu::LoadOp::Clear, .storeOp = wgpu::StoreOp::Store};
    wgpu::RenderPassDescriptor desc {.colorAttachmentCount = 1, .colorAttachments = &color};
    auto pass = encoder.BeginRenderPass(&desc);
    pass.SetPipeline(presentPipeline);
    pass.SetBindGroup(0, presentGroup);
    pass.Draw(3);
    pass.End();
}

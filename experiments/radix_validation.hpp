#pragma once

#include <webgpu/webgpu_cpp.h>
#include <algorithm>
#include <array>
#include <initializer_list>
#include <vector>

struct RadixValidation {
    wgpu::Device device;
    wgpu::ComputePipeline radixHistogramRankPipeline;
    wgpu::ComputePipeline radixScanBlocksPipeline;
    wgpu::ComputePipeline radixScanSumsPipeline;
    wgpu::ComputePipeline radixScatterPipeline;

    wgpu::BindGroup radixHistogramBindGroup;
    wgpu::BindGroup radixScanBlocksBindGroup;
    wgpu::BindGroup radixScanSumsBindGroup;
    wgpu::BindGroup radixScatterBindGroup;

    std::array<wgpu::BindGroup, 8> radixHistogramBindGroups;
    std::array<wgpu::BindGroup, 8> radixScanBlocksBindGroups;
    std::array<wgpu::BindGroup, 8> radixScanSumsBindGroups;
    std::array<wgpu::BindGroup, 8> radixScatterBindGroups;

    wgpu::Buffer gaussianBuffer;
    wgpu::Buffer orderBuffer;
    wgpu::Buffer cameraBuffer;
    wgpu::Buffer radixParamsBuffer;
    wgpu::Buffer radixHistogramBuffer;
    wgpu::Buffer radixBlockOffsetsBuffer;
    wgpu::Buffer radixLocalRanksBuffer;
    std::array<wgpu::Buffer, 2> radixKeysBuffer;

    uint32_t gaussianCount;
    uint32_t radixGroupCount;
    uint32_t radixBlockCount;

    void createRadixPipelines(const wgpu::ShaderModule& shader);
    void createRadixBuffers();
    void recordRadix(const wgpu::CommandEncoder& encoder, const wgpu::PassTimestampWrites* timestamps = nullptr) const;
};

inline void RadixValidation::createRadixPipelines(const wgpu::ShaderModule& shader) {
    auto pipeline = [&](const char* entry) {
        wgpu::ComputePipelineDescriptor desc {
            .label = entry,
            .compute = {.module = shader, .entryPoint = entry},
        };
        return device.CreateComputePipeline(&desc);
    };
    radixHistogramRankPipeline = pipeline("radix_histogram_rank");
    radixScanBlocksPipeline = pipeline("radix_scan_blocks");
    radixScanSumsPipeline = pipeline("radix_scan_sums");
    radixScatterPipeline = pipeline("radix_scatter");
}

inline void RadixValidation::createRadixBuffers() {
    radixGroupCount = (gaussianCount + 255) / 256;
    radixBlockCount = (radixGroupCount + 255) / 256;
    auto buffer = [&](uint64_t bytes, wgpu::BufferUsage usage = wgpu::BufferUsage::Storage) {
        wgpu::BufferDescriptor desc {.usage = usage, .size = std::max(uint64_t(16), bytes)};
        return device.CreateBuffer(&desc);
    };
    for (auto& keyBuffer : radixKeysBuffer) keyBuffer = buffer(uint64_t(gaussianCount) * 8, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    radixLocalRanksBuffer = buffer(uint64_t(gaussianCount) * 4);
    radixHistogramBuffer = buffer(uint64_t(radixGroupCount) * 16 * 4);
    radixBlockOffsetsBuffer = buffer(uint64_t(radixBlockCount) * 16 * 4);
    radixParamsBuffer = buffer(8 * 256, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    for (uint32_t digit = 0; digit < 8; digit++) {
        uint32_t data[] {gaussianCount, radixGroupCount, radixBlockCount, digit * 4};
        device.GetQueue().WriteBuffer(radixParamsBuffer, digit * 256, data, sizeof(data));
    }
    auto scene = [&](const wgpu::ComputePipeline& pipeline, uint32_t count = 0) {
        wgpu::BindGroupEntry entries[] {
            {.binding = 0, .buffer = gaussianBuffer, .size = gaussianBuffer.GetSize()},
            {.binding = 1, .buffer = cameraBuffer, .size = cameraBuffer.GetSize()},
        };
        wgpu::BindGroupDescriptor desc {
            .layout = pipeline.GetBindGroupLayout(0),
            .entryCount = count,
            .entries = entries,
        };
        return device.CreateBindGroup(&desc);
    };
    radixHistogramBindGroup = scene(radixHistogramRankPipeline, 2);
    radixScanBlocksBindGroup = scene(radixScanBlocksPipeline);
    radixScanSumsBindGroup = scene(radixScanSumsPipeline);
    radixScatterBindGroup = scene(radixScatterPipeline);

    auto group = [&](
        const wgpu::ComputePipeline& pipeline,
        std::initializer_list<uint32_t> bindings,
        uint32_t digit,
        const wgpu::Buffer& sourceBuffer,
        const wgpu::Buffer& destinationBuffer
    ) {
        wgpu::Buffer buffers[] {radixParamsBuffer, sourceBuffer, destinationBuffer, radixHistogramBuffer, radixBlockOffsetsBuffer, radixLocalRanksBuffer, orderBuffer};
        std::vector<wgpu::BindGroupEntry> entries;
        for (uint32_t binding : bindings) {
            entries.push_back({.binding = binding, .buffer = buffers[binding],
                .offset = binding == 0 ? digit * 256u : 0u,
                .size = binding == 0 ? 16u : buffers[binding].GetSize()});
        }
        wgpu::BindGroupDescriptor desc {.layout = pipeline.GetBindGroupLayout(1),
            .entryCount = uint32_t(entries.size()), .entries = entries.data()};
        return device.CreateBindGroup(&desc);
    };
    for (uint32_t digit = 0; digit < 8; digit++) {
        const auto& sourceBuffer = radixKeysBuffer[digit % 2];
        const auto& destinationBuffer = radixKeysBuffer[1 - digit % 2];
        radixHistogramBindGroups[digit] = group(radixHistogramRankPipeline, {0, 1, 2, 3, 5}, digit,
            digit == 0 ? destinationBuffer : sourceBuffer, digit == 0 ? sourceBuffer : destinationBuffer);
        radixScanBlocksBindGroups[digit] = group(radixScanBlocksPipeline, {0, 3, 4}, digit, sourceBuffer, destinationBuffer);
        radixScanSumsBindGroups[digit] = group(radixScanSumsPipeline, {0, 4}, digit, sourceBuffer, destinationBuffer);
        radixScatterBindGroups[digit] = group(radixScatterPipeline, {0, 1, 2, 3, 4, 5, 6}, digit, sourceBuffer, destinationBuffer);
    }
}

inline void RadixValidation::recordRadix(const wgpu::CommandEncoder& encoder, const wgpu::PassTimestampWrites* timestamps) const {
    wgpu::ComputePassDescriptor desc {.timestampWrites = timestamps};
    auto pass = encoder.BeginComputePass(&desc);
    auto dispatch = [&](
        const wgpu::ComputePipeline& pipeline,
        const wgpu::BindGroup& scene,
        const wgpu::BindGroup& group,
        uint32_t x,
        uint32_t y = 1
    ) {
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, scene);
        pass.SetBindGroup(1, group);
        pass.DispatchWorkgroups(x, y);
    };
    for (uint32_t digit = 0; digit < 8; digit++) {
        dispatch(radixHistogramRankPipeline, radixHistogramBindGroup, radixHistogramBindGroups[digit], radixGroupCount);
        dispatch(radixScanBlocksPipeline, radixScanBlocksBindGroup, radixScanBlocksBindGroups[digit], radixBlockCount, 16);
        dispatch(radixScanSumsPipeline, radixScanSumsBindGroup, radixScanSumsBindGroups[digit], 1);
        dispatch(radixScatterPipeline, radixScatterBindGroup, radixScatterBindGroups[digit], radixGroupCount);
    }
    pass.End();
}


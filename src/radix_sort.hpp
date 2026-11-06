#pragma once

#include <array>
#include <webgpu/webgpu_cpp.h>

constexpr uint32_t radixWorkgroupSize = 256;
constexpr uint32_t radixBitsPerPass = 4;
constexpr uint32_t radixDigitCount = 1u << radixBitsPerPass;
constexpr uint32_t radixPassCount = 32 / radixBitsPerPass;
constexpr uint32_t radixUniformStride = 256;

struct RadixSort {
    wgpu::Device device;
    wgpu::Queue queue;

    wgpu::ComputePipeline radixHistogramRankPipeline;
    wgpu::ComputePipeline radixScanBlocksPipeline;
    wgpu::ComputePipeline radixScanSumsPipeline;
    wgpu::ComputePipeline radixScatterPipeline;

    wgpu::BindGroup radixHistogramBindGroup;
    wgpu::BindGroup radixScanBlocksBindGroup;
    wgpu::BindGroup radixScanSumsBindGroup;
    wgpu::BindGroup radixScatterBindGroup;

    std::array<wgpu::BindGroup, radixPassCount> radixHistogramBindGroups;
    std::array<wgpu::BindGroup, radixPassCount> radixScanBlocksBindGroups;
    std::array<wgpu::BindGroup, radixPassCount> radixScanSumsBindGroups;
    std::array<wgpu::BindGroup, radixPassCount> radixScatterBindGroups;

    wgpu::Buffer radixParamsBuffer;
    wgpu::Buffer radixHistogramBuffer;
    wgpu::Buffer radixBlockOffsetsBuffer;
    wgpu::Buffer radixLocalRanksBuffer;
    std::array<wgpu::Buffer, 2> radixKeysBuffer;

    uint32_t radixGroupCount;
    uint32_t radixBlockCount;
    uint32_t count;

    void init(const wgpu::Device& device, const wgpu::ShaderModule& shader);
    void createPipelines(const wgpu::ShaderModule& shader);
    void createBuffers(uint32_t size);
    void createBindGroups(
        const wgpu::Buffer& gaussianBuffer,
        const wgpu::Buffer& cameraBuffer,
        const wgpu::Buffer& orderBuffer
    );
    void recordCommands(const wgpu::CommandEncoder& encoder, const wgpu::PassTimestampWrites* timestamps = nullptr) const;
};

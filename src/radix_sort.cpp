#include "radix_sort.hpp"

#include <algorithm>
#include <utility>

struct RadixParams {
    uint32_t count;
    uint32_t groupCount;
    uint32_t blockCount;
    uint32_t bitShift;
};

void RadixSort::init(const wgpu::Device& device, const wgpu::ShaderModule& shader) {
    this->device = device;
    queue = device.GetQueue();
    createPipelines(shader);
}

void RadixSort::createPipelines(const wgpu::ShaderModule& shader) {
    wgpu::ComputePipelineDescriptor histogramRankDesc {
        .label = "radix_histogram_rank",
        .compute = {.module = shader, .entryPoint = "radix_histogram_rank"},
    };
    radixHistogramRankPipeline = device.CreateComputePipeline(&histogramRankDesc);

    wgpu::ComputePipelineDescriptor scanBlocksDesc {
        .label = "radix_scan_blocks",
        .compute = {.module = shader, .entryPoint = "radix_scan_blocks"},
    };
    radixScanBlocksPipeline = device.CreateComputePipeline(&scanBlocksDesc);

    wgpu::ComputePipelineDescriptor scanSumsDesc {
        .label = "radix_scan_sums",
        .compute = {.module = shader, .entryPoint = "radix_scan_sums"},
    };
    radixScanSumsPipeline = device.CreateComputePipeline(&scanSumsDesc);

    wgpu::ComputePipelineDescriptor scatterDesc {
        .label = "radix_scatter",
        .compute = {.module = shader, .entryPoint = "radix_scatter"},
    };
    radixScatterPipeline = device.CreateComputePipeline(&scatterDesc);
}

void RadixSort::createBuffers(uint32_t size) {
    count = size;
    radixGroupCount = (count + radixWorkgroupSize - 1) / radixWorkgroupSize;
    radixBlockCount = (radixGroupCount + radixWorkgroupSize - 1) / radixWorkgroupSize;

    wgpu::BufferDescriptor keysBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        .size = std::max<uint64_t>(16, uint64_t(count) * 2 * sizeof(uint32_t)),
    };
    radixKeysBuffer[0] = device.CreateBuffer(&keysBufferDesc);
    radixKeysBuffer[1] = device.CreateBuffer(&keysBufferDesc);

    wgpu::BufferDescriptor localRanksBufferDesc {
        .usage = wgpu::BufferUsage::Storage,
        .size = std::max<uint64_t>(16, uint64_t(count) * sizeof(uint32_t)),
    };
    radixLocalRanksBuffer = device.CreateBuffer(&localRanksBufferDesc);

    wgpu::BufferDescriptor histogramBufferDesc {
        .usage = wgpu::BufferUsage::Storage,
        .size = std::max<uint64_t>(16, uint64_t(radixDigitCount) * radixGroupCount * sizeof(uint32_t)),
    };
    radixHistogramBuffer = device.CreateBuffer(&histogramBufferDesc);

    wgpu::BufferDescriptor blockOffsetsBufferDesc {
        .usage = wgpu::BufferUsage::Storage,
        .size = std::max<uint64_t>(16, uint64_t(radixDigitCount) * radixBlockCount * sizeof(uint32_t)),
    };
    radixBlockOffsetsBuffer = device.CreateBuffer(&blockOffsetsBufferDesc);

    wgpu::BufferDescriptor paramsBufferDesc {
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = radixPassCount * radixUniformStride,
    };
    radixParamsBuffer = device.CreateBuffer(&paramsBufferDesc);

    for (uint32_t passIndex = 0; passIndex < radixPassCount; passIndex++) {
        RadixParams radixParams {
            .count = count,
            .groupCount = radixGroupCount,
            .blockCount = radixBlockCount,
            .bitShift = passIndex * radixBitsPerPass,
        };
        queue.WriteBuffer(radixParamsBuffer, passIndex * radixUniformStride, &radixParams, sizeof(radixParams));
    }
}

void RadixSort::createBindGroups(
    const wgpu::Buffer& gaussianBuffer,
    const wgpu::Buffer& cameraBuffer,
    const wgpu::Buffer& orderBuffer
) {
    wgpu::BindGroupEntry histogramInputEntries[] {
        {.binding = 0, .buffer = gaussianBuffer, .size = gaussianBuffer.GetSize()},
        {.binding = 1, .buffer = cameraBuffer, .size = cameraBuffer.GetSize()},
    };
    wgpu::BindGroupDescriptor histogramBindGroupDesc {
        .layout = radixHistogramRankPipeline.GetBindGroupLayout(0),
        .entryCount = 2,
        .entries = histogramInputEntries,
    };
    radixHistogramBindGroup = device.CreateBindGroup(&histogramBindGroupDesc);

    wgpu::BindGroupDescriptor scanBlocksBindGroupDesc {
        .layout = radixScanBlocksPipeline.GetBindGroupLayout(0),
    };
    radixScanBlocksBindGroup = device.CreateBindGroup(&scanBlocksBindGroupDesc);

    wgpu::BindGroupDescriptor scanSumsBindGroupDesc {
        .layout = radixScanSumsPipeline.GetBindGroupLayout(0),
    };
    radixScanSumsBindGroup = device.CreateBindGroup(&scanSumsBindGroupDesc);

    wgpu::BindGroupDescriptor scatterBindGroupDesc {
        .layout = radixScatterPipeline.GetBindGroupLayout(0),
    };
    radixScatterBindGroup = device.CreateBindGroup(&scatterBindGroupDesc);

    wgpu::Buffer sourceBuffer = radixKeysBuffer[0];
    wgpu::Buffer destinationBuffer = radixKeysBuffer[1];
    for (uint32_t passIndex = 0; passIndex < radixPassCount; passIndex++) {
        wgpu::Buffer histogramInputBuffer = sourceBuffer;
        wgpu::Buffer histogramOutputBuffer = destinationBuffer;
        if (passIndex == 0) {
            histogramInputBuffer = destinationBuffer;
            histogramOutputBuffer = sourceBuffer;
        }

        wgpu::BindGroupEntry paramsEntry {
            .binding = 0,
            .buffer = radixParamsBuffer,
            .offset = passIndex * radixUniformStride,
            .size = sizeof(RadixParams),
        };

        wgpu::BindGroupEntry histogramEntries[] {
            paramsEntry,
            {.binding = 1, .buffer = histogramInputBuffer, .size = histogramInputBuffer.GetSize()},
            {.binding = 2, .buffer = histogramOutputBuffer, .size = histogramOutputBuffer.GetSize()},
            {.binding = 3, .buffer = radixHistogramBuffer, .size = radixHistogramBuffer.GetSize()},
            {.binding = 5, .buffer = radixLocalRanksBuffer, .size = radixLocalRanksBuffer.GetSize()},
        };
        wgpu::BindGroupDescriptor histogramDesc {
            .layout = radixHistogramRankPipeline.GetBindGroupLayout(1),
            .entryCount = 5,
            .entries = histogramEntries,
        };
        radixHistogramBindGroups[passIndex] = device.CreateBindGroup(&histogramDesc);

        wgpu::BindGroupEntry scanBlocksEntries[] {
            paramsEntry,
            {.binding = 3, .buffer = radixHistogramBuffer, .size = radixHistogramBuffer.GetSize()},
            {.binding = 4, .buffer = radixBlockOffsetsBuffer, .size = radixBlockOffsetsBuffer.GetSize()},
        };
        wgpu::BindGroupDescriptor scanBlocksDesc {
            .layout = radixScanBlocksPipeline.GetBindGroupLayout(1),
            .entryCount = 3,
            .entries = scanBlocksEntries,
        };
        radixScanBlocksBindGroups[passIndex] = device.CreateBindGroup(&scanBlocksDesc);

        wgpu::BindGroupEntry scanSumsEntries[] {
            paramsEntry,
            {.binding = 4, .buffer = radixBlockOffsetsBuffer, .size = radixBlockOffsetsBuffer.GetSize()},
        };
        wgpu::BindGroupDescriptor scanSumsDesc {
            .layout = radixScanSumsPipeline.GetBindGroupLayout(1),
            .entryCount = 2,
            .entries = scanSumsEntries,
        };
        radixScanSumsBindGroups[passIndex] = device.CreateBindGroup(&scanSumsDesc);

        wgpu::BindGroupEntry scatterEntries[] {
            paramsEntry,
            {.binding = 1, .buffer = sourceBuffer, .size = sourceBuffer.GetSize()},
            {.binding = 2, .buffer = destinationBuffer, .size = destinationBuffer.GetSize()},
            {.binding = 3, .buffer = radixHistogramBuffer, .size = radixHistogramBuffer.GetSize()},
            {.binding = 4, .buffer = radixBlockOffsetsBuffer, .size = radixBlockOffsetsBuffer.GetSize()},
            {.binding = 5, .buffer = radixLocalRanksBuffer, .size = radixLocalRanksBuffer.GetSize()},
            {.binding = 6, .buffer = orderBuffer, .size = orderBuffer.GetSize()},
        };
        wgpu::BindGroupDescriptor scatterDesc {
            .layout = radixScatterPipeline.GetBindGroupLayout(1),
            .entryCount = 7,
            .entries = scatterEntries,
        };
        radixScatterBindGroups[passIndex] = device.CreateBindGroup(&scatterDesc);

        std::swap(sourceBuffer, destinationBuffer);
    }
}

void RadixSort::recordCommands(const wgpu::CommandEncoder& encoder, const wgpu::PassTimestampWrites* timestamps) const {
    wgpu::ComputePassDescriptor desc {.timestampWrites = timestamps};
    auto pass = encoder.BeginComputePass(&desc);
    for (uint32_t digit = 0; digit < radixPassCount; digit++) {
        pass.SetPipeline(radixHistogramRankPipeline);
        pass.SetBindGroup(0, radixHistogramBindGroup);
        pass.SetBindGroup(1, radixHistogramBindGroups[digit]);
        pass.DispatchWorkgroups(radixGroupCount);

        pass.SetPipeline(radixScanBlocksPipeline);
        pass.SetBindGroup(0, radixScanBlocksBindGroup);
        pass.SetBindGroup(1, radixScanBlocksBindGroups[digit]);
        pass.DispatchWorkgroups(radixBlockCount, radixDigitCount);

        pass.SetPipeline(radixScanSumsPipeline);
        pass.SetBindGroup(0, radixScanSumsBindGroup);
        pass.SetBindGroup(1, radixScanSumsBindGroups[digit]);
        pass.DispatchWorkgroups(1);

        pass.SetPipeline(radixScatterPipeline);
        pass.SetBindGroup(0, radixScatterBindGroup);
        pass.SetBindGroup(1, radixScatterBindGroups[digit]);
        pass.DispatchWorkgroups(radixGroupCount);
    }
    pass.End();
}

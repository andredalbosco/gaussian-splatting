#include "radix_sort.hpp"

#include <algorithm>

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

    wgpu::BufferDescriptor sortBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        .size = std::max<uint64_t>(16, uint64_t(count) * sizeof(uint32_t)),
    };
    for (size_t i = 0; i < radixKeysBuffer.size(); i++) {
        radixKeysBuffer[i] = device.CreateBuffer(&sortBufferDesc);
        radixValuesBuffer[i] = device.CreateBuffer(&sortBufferDesc);
    }

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

void RadixSort::createBindGroups(const wgpu::Buffer& gaussianBuffer, const wgpu::Buffer& cameraBuffer) {
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

    for (uint32_t passIndex = 0; passIndex < radixPassCount; passIndex++) {
        const uint32_t src = passIndex % 2;
        const uint32_t dst = 1 - src;
        const uint32_t histogramDst = passIndex == 0 ? src : dst;
        const auto& histogramInputKeys = radixKeysBuffer[1 - histogramDst];
        const auto& histogramOutputKeys = radixKeysBuffer[histogramDst];
        const auto& histogramOutputValues = radixValuesBuffer[histogramDst];

        wgpu::BindGroupEntry paramsEntry {
            .binding = 0,
            .buffer = radixParamsBuffer,
            .offset = passIndex * radixUniformStride,
            .size = sizeof(RadixParams),
        };

        wgpu::BindGroupEntry histogramEntries[] {
            paramsEntry,
            {.binding = 1, .buffer = histogramInputKeys, .size = histogramInputKeys.GetSize()},
            {.binding = 2, .buffer = histogramOutputKeys, .size = histogramOutputKeys.GetSize()},
            {.binding = 3, .buffer = radixHistogramBuffer, .size = radixHistogramBuffer.GetSize()},
            {.binding = 5, .buffer = radixLocalRanksBuffer, .size = radixLocalRanksBuffer.GetSize()},
            {.binding = 7, .buffer = histogramOutputValues, .size = histogramOutputValues.GetSize()},
        };
        wgpu::BindGroupDescriptor histogramDesc {
            .layout = radixHistogramRankPipeline.GetBindGroupLayout(1),
            .entryCount = 6,
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

        const auto& inputKeys = radixKeysBuffer[src];
        const auto& outputKeys = radixKeysBuffer[dst];
        const auto& inputValues = radixValuesBuffer[src];
        const auto& outputValues = radixValuesBuffer[dst];
        wgpu::BindGroupEntry scatterEntries[] {
            paramsEntry,
            {.binding = 1, .buffer = inputKeys, .size = inputKeys.GetSize()},
            {.binding = 2, .buffer = outputKeys, .size = outputKeys.GetSize()},
            {.binding = 3, .buffer = radixHistogramBuffer, .size = radixHistogramBuffer.GetSize()},
            {.binding = 4, .buffer = radixBlockOffsetsBuffer, .size = radixBlockOffsetsBuffer.GetSize()},
            {.binding = 5, .buffer = radixLocalRanksBuffer, .size = radixLocalRanksBuffer.GetSize()},
            {.binding = 6, .buffer = inputValues, .size = inputValues.GetSize()},
            {.binding = 7, .buffer = outputValues, .size = outputValues.GetSize()},
        };
        wgpu::BindGroupDescriptor scatterDesc {
            .layout = radixScatterPipeline.GetBindGroupLayout(1),
            .entryCount = 8,
            .entries = scatterEntries,
        };
        radixScatterBindGroups[passIndex] = device.CreateBindGroup(&scatterDesc);
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

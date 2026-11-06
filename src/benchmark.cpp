#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

BenchmarkResult measureRendering(
    const wgpu::Instance& instance,
    const wgpu::Device& device,
    const wgpu::RenderPipeline& pipeline,
    const wgpu::BindGroup& bindGroup,
    wgpu::TextureFormat colorFormat,
    Extent2D size,
    uint32_t gaussianCount
) {
    constexpr uint32_t samples = 128;
    wgpu::Queue queue = device.GetQueue();
    wgpu::TextureDescriptor colorDesc {
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {size.width, size.height, 1},
        .format = colorFormat,
    };
    wgpu::Texture colorTexture = device.CreateTexture(&colorDesc);
    wgpu::TextureDescriptor depthDesc {
        .usage = wgpu::TextureUsage::RenderAttachment,
        .size = colorDesc.size,
        .format = wgpu::TextureFormat::Depth24Plus,
    };
    wgpu::Texture depthTexture = device.CreateTexture(&depthDesc);
    wgpu::RenderPassColorAttachment color {
        .view = colorTexture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.05, 0.05, 0.07, 1},
    };
    wgpu::RenderPassDepthStencilAttachment depth {
        .view = depthTexture.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Discard,
        .depthClearValue = 1,
    };
    wgpu::QuerySetDescriptor queryDesc {.type = wgpu::QueryType::Timestamp, .count = 2 * samples};
    wgpu::QuerySet queries = device.CreateQuerySet(&queryDesc);
    wgpu::BufferDescriptor resolveDesc {
        .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
        .size = 2 * samples * sizeof(uint64_t),
    };
    wgpu::Buffer resolve = device.CreateBuffer(&resolveDesc);
    wgpu::BufferDescriptor readbackDesc {
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        .size = resolveDesc.size,
    };
    wgpu::Buffer readback = device.CreateBuffer(&readbackDesc);
    wgpu::RenderPassDescriptor passDesc {
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
        .depthStencilAttachment = &depth,
    };
    auto draw = [&](const wgpu::CommandEncoder& encoder) {
        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.Draw(4, gaussianCount);
        pass.End();
    };
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    for (uint32_t i = 0; i < 32; i++) draw(encoder);
    for (uint32_t i = 0; i < samples; i++) {
        wgpu::PassTimestampWrites writes {
            .querySet = queries,
            .beginningOfPassWriteIndex = 2 * i,
            .endOfPassWriteIndex = 2 * i + 1,
        };
        passDesc.timestampWrites = &writes;
        draw(encoder);
    }
    encoder.ResolveQuerySet(queries, 0, 2 * samples, resolve, 0);
    encoder.CopyBufferToBuffer(resolve, 0, readback, 0, resolveDesc.size);
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    instance.WaitAny(readback.MapAsync(wgpu::MapMode::Read, 0, readback.GetSize(),
        wgpu::CallbackMode::WaitAnyOnly, [](wgpu::MapAsyncStatus, wgpu::StringView) {}), UINT64_MAX);
    const uint64_t* times = static_cast<const uint64_t*>(readback.GetConstMappedRange());
    std::vector<double> gpuTimes(samples);
    for (uint32_t i = 0; i < samples; i++) gpuTimes[i] = (times[2 * i + 1] - times[2 * i]) / 1e6;
    readback.Unmap();
    std::sort(gpuTimes.begin(), gpuTimes.end());

    passDesc.timestampWrites = nullptr;
    encoder = device.CreateCommandEncoder();
    for (uint32_t i = 0; i < samples; i++) draw(encoder);
    commands = encoder.Finish();
    auto start = std::chrono::steady_clock::now();
    queue.Submit(1, &commands);
    instance.WaitAny(queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    double throughputMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / samples;

    uint32_t rowBytes = (size.width * 4 + 255) & ~255u;
    readbackDesc.size = uint64_t(rowBytes) * size.height;
    wgpu::Buffer pixels = device.CreateBuffer(&readbackDesc);
    wgpu::TexelCopyTextureInfo source {.texture = colorTexture};
    wgpu::TexelCopyBufferInfo destination {
        .layout = {.bytesPerRow = rowBytes, .rowsPerImage = size.height},
        .buffer = pixels,
    };
    encoder = device.CreateCommandEncoder();
    encoder.CopyTextureToBuffer(&source, &destination, &colorDesc.size);
    commands = encoder.Finish();
    queue.Submit(1, &commands);
    instance.WaitAny(pixels.MapAsync(wgpu::MapMode::Read, 0, pixels.GetSize(),
        wgpu::CallbackMode::WaitAnyOnly, [](wgpu::MapAsyncStatus, wgpu::StringView) {}), UINT64_MAX);
    BenchmarkResult result {gpuTimes[samples / 2], gpuTimes[samples * 9 / 10], throughputMs, {}};
    result.pixels.resize(size.width * size.height * 4);
    const uint8_t* data = static_cast<const uint8_t*>(pixels.GetConstMappedRange());
    for (uint32_t y = 0; y < size.height; y++) {
        std::memcpy(result.pixels.data() + y * size.width * 4, data + y * rowBytes, size.width * 4);
    }
    pixels.Unmap();
    return result;
}

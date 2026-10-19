#include "camera.hpp"
#include "gaussian.hpp"
#include "glfw_utils.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <Eigen/Geometry>
#include <Eigen/LU>

int main() {
    auto instance = createInstance();
    auto adapter = requestAdapter(instance, {});
    auto device = requestDevice(instance, adapter);
    auto queue = device.GetQueue();
    auto shader = readWgslFile(device, SRC_DIR "/shaders.wgsl");
    wgpu::BlendState blend {
        .color = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
    };
    wgpu::ColorTargetState target {.format = wgpu::TextureFormat::RGBA8Unorm, .blend = &blend};
    wgpu::FragmentState fragment {.module = shader, .entryPoint = "fs_main", .targetCount = 1, .targets = &target};
    wgpu::RenderPipelineDescriptor pipelineDesc {
        .vertex = {.module = shader, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleStrip}, .fragment = &fragment,
    };
    auto pipeline = device.CreateRenderPipeline(&pipelineDesc);
    int failures = 0;
    for (int test = 0; test < 5; test++) {
        uint32_t width = test == 3 ? 257 : 129, height = test == 3 ? 97 : 129;
        Eigen::Matrix4f vp = projectionMatrix(60, float(width) / height, 0.1f, 20);
        std::vector<Gaussian> shapes {
            {{0, 0, 2, 1}, {0.2f, 0.2f, 0.2f, 0}, {0, 0, 0, 1}, {1, 0, 0, 0.5f}},
        };
        if (test == 1) shapes.push_back({{0, 0, 3, 1}, {0.3f, 0.3f, 0.3f, 0}, {0, 0, 0, 1}, {0, 0, 1, 0.5f}});
        if (test == 2 || test == 3) {
            shapes[0].position = {0.35f, -0.15f, 2, 1};
            shapes[0].scale = {0.4f, 0.05f, 0.2f, 0};
            Eigen::Quaternionf q = Eigen::AngleAxisf(0.7f, Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(0.9f, Eigen::Vector3f::UnitY());
            shapes[0].rotation = q.coeffs();
        }
        if (test == 4) {
            shapes[0].position.z() = -2;
            shapes.push_back({{0, 0, 0.05f, 1}, {0.2f, 0.2f, 0.2f, 0}, {0, 0, 0, 1}, {0, 1, 0, 0.9f}});
            shapes.push_back({{0, 0, 30, 1}, {2, 2, 2, 0}, {0, 0, 0, 1}, {0, 0, 1, 0.9f}});
        }
        std::vector<uint32_t> order(shapes.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return shapes[a].position.z() > shapes[b].position.z(); });
        wgpu::BufferDescriptor bufferDesc {.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                          .size = shapes.size() * sizeof(Gaussian)};
        auto shapeBuffer = device.CreateBuffer(&bufferDesc);
        queue.WriteBuffer(shapeBuffer, 0, shapes.data(), bufferDesc.size);
        bufferDesc.size = order.size() * sizeof(uint32_t);
        auto orderBuffer = device.CreateBuffer(&bufferDesc);
        queue.WriteBuffer(orderBuffer, 0, order.data(), bufferDesc.size);
        bufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        bufferDesc.size = 80;
        auto cameraBuffer = device.CreateBuffer(&bufferDesc);
        Eigen::Vector4f viewport {float(width), float(height), 0, 0};
        queue.WriteBuffer(cameraBuffer, 0, vp.data(), 64);
        queue.WriteBuffer(cameraBuffer, 64, viewport.data(), 16);
        wgpu::BindGroupEntry entries[] {
            {.binding = 0, .buffer = shapeBuffer, .size = shapeBuffer.GetSize()},
            {.binding = 1, .buffer = cameraBuffer, .size = 80},
            {.binding = 2, .buffer = orderBuffer, .size = orderBuffer.GetSize()},
        };
        wgpu::BindGroupDescriptor groupDesc {.layout = pipeline.GetBindGroupLayout(0), .entryCount = 3, .entries = entries};
        auto group = device.CreateBindGroup(&groupDesc);
        wgpu::TextureDescriptor textureDesc {.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
            .size = {width, height, 1}, .format = target.format};
        auto texture = device.CreateTexture(&textureDesc);
        wgpu::RenderPassColorAttachment color {.view = texture.CreateView(), .loadOp = wgpu::LoadOp::Clear,
                                              .storeOp = wgpu::StoreOp::Store, .clearValue = {0, 0, 0, 0}};
        wgpu::RenderPassDescriptor passDesc {.colorAttachmentCount = 1, .colorAttachments = &color};
        auto encoder = device.CreateCommandEncoder();
        auto pass = encoder.BeginRenderPass(&passDesc);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, group);
        pass.Draw(4, shapes.size());
        pass.End();
        uint32_t rowBytes = (width * 4 + 255) & ~255u;
        bufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        bufferDesc.size = rowBytes * height;
        auto readback = device.CreateBuffer(&bufferDesc);
        wgpu::TexelCopyTextureInfo source {.texture = texture};
        wgpu::TexelCopyBufferInfo destination {.layout = {.bytesPerRow = rowBytes, .rowsPerImage = height}, .buffer = readback};
        encoder.CopyTextureToBuffer(&source, &destination, &textureDesc.size);
        auto commands = encoder.Finish();
        queue.Submit(1, &commands);
        instance.WaitAny(readback.MapAsync(wgpu::MapMode::Read, 0, readback.GetSize(), wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus, wgpu::StringView) {}), UINT64_MAX);
        auto* pixels = static_cast<const uint8_t*>(readback.GetConstMappedRange());
        std::vector<Eigen::Vector4d> expected(width * height, Eigen::Vector4d::Zero());
        for (uint32_t index : order) {
            const auto& shape = shapes[index];
            Eigen::Matrix4d matrix = vp.cast<double>();
            Eigen::Vector3d mean = shape.position.head<3>().cast<double>();
            Eigen::Vector4d clip = matrix * Eigen::Vector4d(mean.x(), mean.y(), mean.z(), 1);
            if (clip.w() <= 0 || clip.z() < 0 || clip.z() > clip.w()) continue;
            auto project = [&](const Eigen::Vector3d& point) -> Eigen::Vector2d {
                Eigen::Vector4d p = matrix * Eigen::Vector4d(point.x(), point.y(), point.z(), 1);
                return (p.head<2>() / p.w()).cwiseProduct(Eigen::Vector2d(width * 0.5, height * 0.5));
            };
            Eigen::Quaterniond rotation(shape.rotation.w(), shape.rotation.x(), shape.rotation.y(), shape.rotation.z());
            Eigen::Matrix<double, 2, 3> axes;
            for (int axis = 0; axis < 3; axis++) {
                Eigen::Vector3d delta = rotation * (Eigen::Vector3d::Unit(axis) * shape.scale[axis]);
                axes.col(axis) = (project(mean + 1e-4 * delta) - project(mean - 1e-4 * delta)) / 2e-4;
            }
            Eigen::Matrix2d covariance = axes * axes.transpose() + 0.3 * Eigen::Matrix2d::Identity();
            Eigen::Matrix2d inverse = covariance.inverse();
            for (uint32_t y = 0; y < height; y++) for (uint32_t x = 0; x < width; x++) {
                Eigen::Vector2d delta = Eigen::Vector2d(x + 0.5 - width * 0.5, height * 0.5 - y - 0.5) - project(mean);
                double q = delta.dot(inverse * delta);
                if (q > 9) continue;
                double alpha = std::min(0.99, shape.color.w() * std::exp(-0.5 * q));
                if (alpha < 1.0 / 255) continue;
                Eigen::Vector4d value;
                value << shape.color.head<3>().cast<double>() * alpha, alpha;
                expected[y * width + x] = value + (1 - alpha) * expected[y * width + x];
            }
        }
        int maxError = 0;
        size_t changed = 0;
        for (uint32_t y = 0; y < height; y++) for (uint32_t x = 0; x < width; x++) for (int channel = 0; channel < 4; channel++) {
            int error = std::abs(int(pixels[y * rowBytes + 4 * x + channel]) - int(std::round(expected[y * width + x][channel] * 255)));
            maxError = std::max(maxError, error);
            changed += error > 2;
        }
        if (test == 1) {
            const auto* center = pixels + (height / 2) * rowBytes + (width / 2) * 4;
            failures += std::abs(int(center[0]) - 128) > 1 || std::abs(int(center[2]) - 64) > 1;
        }
        failures += changed != 0;
        readback.Unmap();
        std::cout << "case " << test << ": max channel error " << maxError << "/255, channels beyond tolerance " << changed << '\n';
    }
    return failures ? 1 : 0;
}

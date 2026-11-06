#include "camera.hpp"
#include "colored_gaussian.hpp"
#include "radix_sort.hpp"
#include "glfw_utils.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <numbers>
#include <fstream>
#include <filesystem>
#include <Eigen/Geometry>
#include <Eigen/LU>

double shBasis(int index, const Eigen::Vector3d& direction) {
    int degree = int(std::sqrt(index));
    int order = index - degree * degree - degree;
    int m = std::abs(order);
    double z = direction.z();
    double p = 1;
    for (int i = 1; i <= m; i++) p *= -(2 * i - 1) * std::sqrt(std::max(0.0, 1 - z * z));
    double previous = 0;
    for (int l = m + 1; l <= degree; l++) {
        double next = ((2 * l - 1) * z * p - (l + m - 1) * previous) / (l - m);
        previous = p;
        p = next;
    }
    double normalization = std::sqrt((2 * degree + 1) / (4 * std::numbers::pi) *
        std::tgamma(degree - m + 1) / std::tgamma(degree + m + 1));
    double phi = std::atan2(direction.y(), direction.x());
    if (order < 0) return std::sqrt(2.0) * normalization * p * std::sin(m * phi);
    if (order > 0) return std::sqrt(2.0) * normalization * p * std::cos(m * phi);
    return normalization * p;
}

int main() {
    auto instance = createInstance();
    auto adapter = requestAdapter(instance, {});
    auto device = requestDevice(instance, adapter);
    auto queue = device.GetQueue();
    auto shader = readWgslFile(device, SRC_DIR "/shaders.wgsl");
    RadixSort radixSort;
    radixSort.init(device, shader);
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
    auto fixture = std::filesystem::temp_directory_path() / "gaussian-sh-validation.ply";
    for (int degree = 0; degree <= 3; degree++) {
        int count = (degree + 1) * (degree + 1);
        std::ofstream file(fixture, std::ios::binary);
        file << "ply\nformat binary_little_endian 1.0\nelement vertex 1\n";
        for (const char* name : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) {
            file << "property float " << name << '\n';
        }
        for (int i = 0; i < 3 * (count - 1); i++) file << "property float f_rest_" << i << '\n';
        file << "end_header\n";
        std::vector<float> row {0, 0, 2, 0.1f, 0.2f, 0.3f, 0, -2, -2, -2, 1, 0, 0, 0};
        for (int channel = 0; channel < 3; channel++) for (int i = 1; i < count; i++) {
            row.push_back(float(100 * channel + i));
        }
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        file.close();
        auto gaussians = readPlyFile(fixture.c_str());
        failures += gaussians.size() != 1;
        for (int channel = 0; channel < 3; channel++) {
            failures += gaussians[0].sh.coefficients[0][channel] != row[3 + channel];
            for (int i = 1; i < 16; i++) {
                failures += gaussians[0].sh.coefficients[i][channel] != (i < count ? float(100 * channel + i) : 0.0f);
            }
        }
    }
    std::filesystem::remove(fixture);
    auto jacket = readPlyFile(ASSETS_DIR "/black_jacket.ply");
    failures += jacket.size() != 41470;
    auto luigi = readPlyFile(ASSETS_DIR "/luigi.ply");
    failures += luigi.size() != 14526;
    for (int test = 0; test < 95; test++) {
        uint32_t width = test == 3 ? 257 : 129, height = test == 3 ? 97 : 129;
        Eigen::Matrix4f projection = projectionMatrix(60, float(width) / height, 0.1f, 20);
        Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
        if (test == 5) {
            view.block<3, 3>(0, 0) = (Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(-0.2f, Eigen::Vector3f::UnitX())).toRotationMatrix();
            view.block<3, 1>(0, 3) = Eigen::Vector3f(0.1f, -0.2f, 0.1f);
        }
        Eigen::Matrix4f vp = projection * view;
        std::vector<ColoredGaussian> shapes {
            {{0, 0, 2, 1}, {0.2f, 0.2f, 0.2f, 0}, {0, 0, 0, 1}, {1, 0, 0, 0.5f}},
        };
        int coefficient = test < 80 ? test / 5 : 0;
        Eigen::Vector3d amplitude = test < 80 ? Eigen::Vector3d(0.2, 0.4, -0.3) : Eigen::Vector3d(5, -5, 2);
        Eigen::Vector3d directions[] {{0.3, 0.4, 0.8}, {-0.6, 0.2, -0.5},
            {0.1, -0.7, 0.4}, {-0.4, -0.3, 0.9}, {0.8, 0.6, -0.3}};
        Eigen::Vector3d direction = directions[test % 5].normalized();
        Eigen::Vector3f eye = shapes[0].position.head<3>() - 3 * direction.cast<float>();
        view = viewMatrix(eye, shapes[0].position.head<3>(), Eigen::Vector3f::UnitY());
        vp = projection * view;
        SHCoefficients sh;
        sh.coefficients[coefficient] = {float(amplitude.x()), float(amplitude.y()), float(amplitude.z()), 0};
        if (test >= 85 && test < 90) sh = jacket[(test * 791) % jacket.size()].sh;
        if (test >= 90) sh = luigi[(test * 791) % luigi.size()].sh;
        shapes[0].color.head<3>().setConstant(0.5f);
        std::vector<uint32_t> order(shapes.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return shapes[a].position.z() > shapes[b].position.z(); });
        auto gpuShapes = toGaussians(shapes);
        gpuShapes[0].sh = sh;
        wgpu::BufferDescriptor bufferDesc {.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                          .size = gpuShapes.size() * sizeof(Gaussian)};
        auto shapeBuffer = device.CreateBuffer(&bufferDesc);
        queue.WriteBuffer(shapeBuffer, 0, gpuShapes.data(), bufferDesc.size);
        bufferDesc.size = order.size() * sizeof(uint32_t);
        auto orderBuffer = device.CreateBuffer(&bufferDesc);
        bufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        bufferDesc.size = 144;
        auto cameraBuffer = device.CreateBuffer(&bufferDesc);
        Eigen::Vector4f viewport {float(width), float(height), projection(0, 0), projection(1, 1)};
        queue.WriteBuffer(cameraBuffer, 0, vp.data(), 64);
        queue.WriteBuffer(cameraBuffer, 64, view.data(), 64);
        queue.WriteBuffer(cameraBuffer, 128, viewport.data(), 16);
        radixSort.createBuffers(shapes.size());
        radixSort.createBindGroups(shapeBuffer, cameraBuffer, orderBuffer);
        wgpu::BindGroupEntry entries[] {
            {.binding = 0, .buffer = shapeBuffer, .size = shapeBuffer.GetSize()},
            {.binding = 1, .buffer = cameraBuffer, .size = cameraBuffer.GetSize()},
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
        radixSort.recordCommands(encoder);
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
        Eigen::Vector3d rgb = Eigen::Vector3d::Constant(0.5);
        for (int i = 0; i < 16; i++) for (int channel = 0; channel < 3; channel++) {
            rgb[channel] += shBasis(i, direction) * sh.coefficients[i][channel];
        }
        shapes[0].color.head<3>() = rgb.cwiseMax(0.0).cast<float>();
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
            int error = std::abs(int(pixels[y * rowBytes + 4 * x + channel]) - int(std::round(std::clamp(expected[y * width + x][channel], 0.0, 1.0) * 255)));
            maxError = std::max(maxError, error);
            changed += error > 2;
        }
        failures += changed != 0;
        readback.Unmap();
        std::cout << "case " << test << ": max channel error " << maxError << "/255, channels beyond tolerance " << changed << '\n';
    }
    return failures ? 1 : 0;
}

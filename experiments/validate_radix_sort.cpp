#include "camera.hpp"
#include "gaussian.hpp"
#include "radix_sort.hpp"
#include "wgpu_utils.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

struct CameraData { Eigen::Matrix4f vp, view; Eigen::Vector4f viewport; };
struct Key { uint32_t depth, index; };

int main(int argc, char** argv) {
    auto instance = createInstance();
    auto adapter = requestAdapter(instance, {});
    wgpu::FeatureName feature = wgpu::FeatureName::TimestampQuery;
    const char* disabled[] {"timestamp_quantization"};
    wgpu::DawnTogglesDescriptor toggles;
    toggles.disabledToggleCount = 1; toggles.disabledToggles = disabled;
    wgpu::Limits limits {};
    limits.maxStorageBufferBindingSize = 320 * 1024 * 1024;
    limits.maxBufferSize = 320 * 1024 * 1024;
    wgpu::DeviceDescriptor desc {};
    desc.requiredLimits = &limits;
    desc.nextInChain = &toggles; desc.requiredFeatureCount = 1; desc.requiredFeatures = &feature;
    desc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
        std::cerr << std::string_view(message.data, message.length) << '\n'; std::exit(2);
    });
    wgpu::Device device;
    instance.WaitAny(adapter.RequestDevice(&desc, wgpu::CallbackMode::WaitAnyOnly,
        [&](wgpu::RequestDeviceStatus, wgpu::Device result, wgpu::StringView) { device = result; }), UINT64_MAX);
    auto queue = device.GetQueue();
    auto shader = readWgslFile(device, SRC_DIR "/shaders.wgsl");
    RadixSort sorter;
    sorter.init(device, shader);
    auto buffer = [&](uint64_t bytes, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor d {.usage = usage, .size = std::max(uint64_t(64), bytes)};
        return device.CreateBuffer(&d);
    };
    auto submit = [&](const wgpu::CommandEncoder& encoder) {
        auto commands = encoder.Finish(); queue.Submit(1, &commands);
    };
    auto read = [&](const wgpu::Buffer& source, uint64_t bytes) {
        auto staging = buffer(bytes, wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead);
        auto encoder = device.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(source, 0, staging, 0, bytes); submit(encoder);
        instance.WaitAny(staging.MapAsync(wgpu::MapMode::Read, 0, bytes, wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus status, wgpu::StringView) { if (status != wgpu::MapAsyncStatus::Success) std::exit(2); }), UINT64_MAX);
        std::vector<uint8_t> result(bytes);
        std::memcpy(result.data(), staging.GetConstMappedRange(0, bytes), bytes); staging.Unmap();
        return result;
    };
    CameraData camera {Eigen::Matrix4f::Identity(), Eigen::Matrix4f::Identity(), {1920, 1080, 1, 1}};
    auto cameraBuffer = buffer(sizeof(camera), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    std::mt19937 rng(20261019);
    std::uniform_real_distribution<float> random(-100, 100);
    auto makeData = [&](uint32_t count, uint32_t pattern) {
        std::vector<Gaussian> data(count);
        for (uint32_t i = 0; i < count; i++) {
            float depth = random(rng);
            if (pattern == 1) depth = 3;
            if (pattern == 2) depth = float(i);
            if (pattern == 3) depth = -float(i);
            if (pattern == 4) depth = float(int(i % 17) - 8);
            if (pattern == 5) {
                const float values[] {0.0f, -0.0f, 1e30f, -1e30f, 1e-30f, -1e-30f, 1, -1};
                depth = values[i % 8];
            }
            data[i] = {{random(rng), random(rng), depth}, 0.5f, {0.1f, 0.1f, 0.1f, 0}, {0, 0, 0, 1}};
        }
        return data;
    };
    auto setup = [&](const std::vector<Gaussian>& data) {
        sorter.radixHistogramBindGroup = nullptr;
        sorter.radixHistogramBindGroups = {};
        sorter.radixScanBlocksBindGroups = {};
        sorter.radixScanSumsBindGroups = {};
        sorter.radixScatterBindGroups = {};
        auto input = buffer(std::max(size_t(sizeof(Gaussian)), data.size() * sizeof(Gaussian)), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
        auto output = buffer(data.size() * 4, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
        if (!data.empty()) queue.WriteBuffer(input, 0, data.data(), data.size() * sizeof(Gaussian));
        sorter.createBuffers(data.size());
        sorter.createBindGroups(input, cameraBuffer, output);
        return std::pair {input, output};
    };
    uint32_t maxCount = std::numeric_limits<uint32_t>::max();
    if (const char* limit = std::getenv("RADIX_TEST_MAX_COUNT")) maxCount = std::strtoul(limit, nullptr, 10);
    int failures = 0;
    auto test = [&](uint32_t count, uint32_t pattern, bool rotated) {
        if (count > maxCount) {
            std::cout << "skipped count=" << count << " (RADIX_TEST_MAX_COUNT)" << std::endl;
            return;
        }
        auto data = makeData(count, pattern);
        auto [input, output] = setup(data);
        camera.view = Eigen::Matrix4f::Identity();
        if (rotated) camera.view.block<3, 3>(0, 0) = (Eigen::AngleAxisf(0.4f, Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(0.7f, Eigen::Vector3f::UnitX())).toRotationMatrix();
        queue.WriteBuffer(cameraBuffer, 0, &camera, sizeof(camera));
        auto encoder = device.CreateCommandEncoder();
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(sorter.radixHistogramRankPipeline); pass.SetBindGroup(0, sorter.radixHistogramBindGroup);
        pass.SetBindGroup(1, sorter.radixHistogramBindGroups[0]); pass.DispatchWorkgroups(sorter.radixGroupCount); pass.End();
        submit(encoder);
        std::vector<uint8_t> original;
        if (count) original = read(sorter.radixKeysBuffer[0], uint64_t(count) * sizeof(Key));
        auto keys = reinterpret_cast<const Key*>(original.data());
        std::vector<uint32_t> expected(count);
        std::iota(expected.begin(), expected.end(), 0u);
        std::sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) {
            return keys[a].depth == keys[b].depth ? a < b : keys[a].depth < keys[b].depth;
        });
        uint32_t errors = 0;
        for (uint32_t i = 0; i < count; i++) {
            float depth = camera.view.block<1, 3>(2, 0).dot(data[i].position.head<3>());
            uint32_t decoded = ~keys[i].depth;
            decoded ^= (decoded & 0x80000000u) ? 0x80000000u : 0xffffffffu;
            float gpuDepth = std::bit_cast<float>(decoded);
            float magnitude = camera.view.block<1, 3>(2, 0).cwiseAbs().dot(data[i].position.head<3>().cwiseAbs());
            float tolerance = 4 * std::numeric_limits<float>::epsilon() * (1 + magnitude);
            errors += keys[i].index != i || std::abs(gpuDepth - depth) > tolerance;
            if (!rotated) {
                uint32_t bits = std::bit_cast<uint32_t>(depth == 0 ? 0.0f : depth);
                uint32_t key = ~(bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u));
                errors += key != keys[i].depth;
            }
        }
        for (int repeat = 0; repeat < 2; repeat++) {
            encoder = device.CreateCommandEncoder(); sorter.recordCommands(encoder); submit(encoder);
            if (!count) continue;
            auto bytes = read(output, uint64_t(count) * 4);
            auto actual = reinterpret_cast<const uint32_t*>(bytes.data());
            for (uint32_t i = 0; i < count; i++) errors += actual[i] != expected[i];
        }
        failures += errors != 0;
        std::cout << "count=" << count << " pattern=" << pattern << " rotated=" << rotated << " errors=" << errors << std::endl;
    };
    for (uint32_t count : {0u, 1u, 2u, 15u, 16u, 17u, 255u, 256u, 257u, 511u, 512u, 513u,
                           65535u, 65536u, 65537u, 100000u, 1000000u, 1048577u}) test(count, 0, false);
    for (uint32_t pattern = 1; pattern <= 5; pattern++) test(65537, pattern, false);
    test(100000, 0, true);
    test(257, 1, true);
    if (failures || argc < 2) return failures ? 1 : 0;

    std::filesystem::path out = argv[1]; std::filesystem::create_directories(out);
    std::ofstream results(out / "timings.csv");
    results << "count,cpu_depth_sort_ms,gpu_depth_sort_ms,gpu_p90_ms\n";
    constexpr uint32_t samples = 32;
    for (uint32_t count : {10000u, 100000u, 1000000u}) {
        auto data = makeData(count, 0);
        auto [input, output] = setup(data);
        camera.view = Eigen::Matrix4f::Identity(); queue.WriteBuffer(cameraBuffer, 0, &camera, sizeof(camera));
        auto encoder = device.CreateCommandEncoder();
        for (int warm = 0; warm < 4; warm++) sorter.recordCommands(encoder);
        submit(encoder);
        wgpu::QuerySetDescriptor qd {.type = wgpu::QueryType::Timestamp, .count = samples * 2};
        auto queries = device.CreateQuerySet(&qd);
        auto resolve = buffer(samples * 16, wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc);
        encoder = device.CreateCommandEncoder();
        for (uint32_t sample = 0; sample < samples; sample++) {
            wgpu::PassTimestampWrites writes {.querySet = queries, .beginningOfPassWriteIndex = sample * 2,
                                              .endOfPassWriteIndex = sample * 2 + 1};
            sorter.recordCommands(encoder, &writes);
        }
        encoder.ResolveQuerySet(queries, 0, samples * 2, resolve, 0); submit(encoder);
        auto bytes = read(resolve, samples * 16);
        auto times = reinterpret_cast<const uint64_t*>(bytes.data());
        std::vector<double> gpu(samples);
        for (uint32_t i = 0; i < samples; i++) gpu[i] = (times[i * 2 + 1] - times[i * 2]) / 1e6;
        std::sort(gpu.begin(), gpu.end());
        std::vector<double> cpu;
        std::vector<uint32_t> order(count); std::vector<float> depths(count);
        for (int trial = 0; trial < 5; trial++) {
            std::iota(order.begin(), order.end(), 0u);
            auto begin = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < count; i++) depths[i] = camera.view.block<1, 3>(2, 0).dot(data[i].position.head<3>());
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                return depths[a] == depths[b] ? a < b : depths[a] > depths[b];
            });
            cpu.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        }
        std::sort(cpu.begin(), cpu.end());
        results << count << ',' << cpu[2] << ',' << gpu[samples / 2] << ',' << gpu[samples * 9 / 10] << '\n';
        std::cout << "benchmark count=" << count << " CPU-ms=" << cpu[2] << " GPU-ms=" << gpu[samples / 2] << '\n';
    }
    return 0;
}

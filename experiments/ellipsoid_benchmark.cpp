#include "camera.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <Eigen/Geometry>
#include <Eigen/LU>

constexpr uint32_t samples = 64;
constexpr auto colorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto depthFormat = wgpu::TextureFormat::Depth32Float;

struct Variant {
    std::string name;
    wgpu::RenderPipeline pipeline;
    uint32_t vertices;
    bool approximate = false;
};
struct Result {
    std::vector<double> times;
    double batchMs;
    std::vector<uint8_t> color;
    std::vector<float> depth;
};
struct Scene {
    std::string name;
    std::vector<Gaussian> shapes;
    OrbitCamera camera;
};

Result measure(const wgpu::Instance& instance, const wgpu::Device& device,
               const Variant& variant, const wgpu::BindGroup& group,
               Extent2D size, uint32_t instances) {
    auto queue = device.GetQueue();
    wgpu::TextureDescriptor textureDesc {
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
        .size = {size.width, size.height, 1}, .format = colorFormat,
    };
    auto colorTexture = device.CreateTexture(&textureDesc);
    textureDesc.format = depthFormat;
    auto depthTexture = device.CreateTexture(&textureDesc);
    wgpu::RenderPassColorAttachment color {.view = colorTexture.CreateView(), .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store, .clearValue = {0.035, 0.045, 0.065, 1}};
    wgpu::RenderPassDepthStencilAttachment depth {.view = depthTexture.CreateView(), .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store, .depthClearValue = 1};
    wgpu::RenderPassDescriptor passDesc {.colorAttachmentCount = 1, .colorAttachments = &color, .depthStencilAttachment = &depth};
    wgpu::QuerySetDescriptor queryDesc {.type = wgpu::QueryType::Timestamp, .count = samples * 2};
    auto queries = device.CreateQuerySet(&queryDesc);
    wgpu::BufferDescriptor bufferDesc {.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
                                      .size = samples * 2 * sizeof(uint64_t)};
    auto resolve = device.CreateBuffer(&bufferDesc);
    bufferDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    auto timestamps = device.CreateBuffer(&bufferDesc);
    auto draw = [&](const wgpu::CommandEncoder& encoder) {
        auto pass = encoder.BeginRenderPass(&passDesc);
        if (variant.vertices) {
            pass.SetPipeline(variant.pipeline);
            pass.SetBindGroup(0, group);
            pass.Draw(variant.vertices, instances);
        }
        pass.End();
    };
    auto encoder = device.CreateCommandEncoder();
    for (int i = 0; i < 16; i++) draw(encoder);
    auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    instance.WaitAny(queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    encoder = device.CreateCommandEncoder();
    for (uint32_t i = 0; i < samples; i++) {
        wgpu::PassTimestampWrites writes {.querySet = queries, .beginningOfPassWriteIndex = 2 * i,
                                         .endOfPassWriteIndex = 2 * i + 1};
        passDesc.timestampWrites = &writes;
        draw(encoder);
    }
    encoder.ResolveQuerySet(queries, 0, samples * 2, resolve, 0);
    encoder.CopyBufferToBuffer(resolve, 0, timestamps, 0, bufferDesc.size);
    commands = encoder.Finish();
    queue.Submit(1, &commands);
    instance.WaitAny(timestamps.MapAsync(wgpu::MapMode::Read, 0, timestamps.GetSize(), wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus, wgpu::StringView) {}), UINT64_MAX);
    auto* values = static_cast<const uint64_t*>(timestamps.GetConstMappedRange());
    Result result;
    for (uint32_t i = 0; i < samples; i++) result.times.push_back((values[2 * i + 1] - values[2 * i]) / 1e6);
    timestamps.Unmap();
    passDesc.timestampWrites = nullptr;
    encoder = device.CreateCommandEncoder();
    for (uint32_t i = 0; i < samples; i++) draw(encoder);
    commands = encoder.Finish();
    auto start = std::chrono::steady_clock::now();
    queue.Submit(1, &commands);
    instance.WaitAny(queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    result.batchMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / samples;

    uint32_t rowBytes = (size.width * 4 + 255) & ~255u;
    bufferDesc.size = uint64_t(rowBytes) * size.height * 2;
    auto readback = device.CreateBuffer(&bufferDesc);
    encoder = device.CreateCommandEncoder();
    wgpu::TexelCopyTextureInfo source {.texture = colorTexture};
    wgpu::TexelCopyBufferInfo destination {.layout = {.bytesPerRow = rowBytes, .rowsPerImage = size.height}, .buffer = readback};
    encoder.CopyTextureToBuffer(&source, &destination, &textureDesc.size);
    source.texture = depthTexture;
    source.aspect = wgpu::TextureAspect::DepthOnly;
    destination.layout.offset = uint64_t(rowBytes) * size.height;
    encoder.CopyTextureToBuffer(&source, &destination, &textureDesc.size);
    commands = encoder.Finish();
    queue.Submit(1, &commands);
    instance.WaitAny(readback.MapAsync(wgpu::MapMode::Read, 0, readback.GetSize(), wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus, wgpu::StringView) {}), UINT64_MAX);
    auto* bytes = static_cast<const uint8_t*>(readback.GetConstMappedRange());
    result.color.resize(size_t(size.width) * size.height * 4);
    result.depth.resize(size_t(size.width) * size.height);
    for (uint32_t y = 0; y < size.height; y++) {
        std::memcpy(result.color.data() + size_t(y) * size.width * 4, bytes + size_t(y) * rowBytes, size.width * 4);
        std::memcpy(result.depth.data() + size_t(y) * size.width,
                    bytes + size_t(rowBytes) * size.height + size_t(y) * rowBytes, size.width * 4);
    }
    readback.Unmap();
    return result;
}

int main(int argc, char* argv[]) {
    std::filesystem::path output = argc > 1 ? argv[1] : "build/ellipsoid-benchmark";
    std::filesystem::create_directories(output);
    auto instance = createInstance();
    auto adapter = requestAdapter(instance, {});
    wgpu::AdapterInfo info;
    adapter.GetInfo(&info);
    std::ofstream metadata(output / "adapter.txt");
    metadata << "device: " << std::string_view(info.device.data, info.device.length)
             << "\ndescription: " << std::string_view(info.description.data, info.description.length)
             << "\nbackend enum: " << int(info.backendType) << "\n64 samples, 16 warmup passes, 3 shuffled trials; timestamp_quantization disabled\n";
    std::cerr << "GPU: " << std::string_view(info.device.data, info.device.length) << '\n';
    wgpu::FeatureName feature = wgpu::FeatureName::TimestampQuery;
    const char* disabled[] {"timestamp_quantization"};
    wgpu::DawnTogglesDescriptor toggles;
    toggles.disabledToggleCount = 1;
    toggles.disabledToggles = disabled;
    wgpu::DeviceDescriptor deviceDesc {};
    deviceDesc.nextInChain = &toggles;
    deviceDesc.requiredFeatureCount = 1;
    deviceDesc.requiredFeatures = &feature;
    deviceDesc.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
        std::cerr << std::string_view(message.data, message.length) << '\n';
        std::exit(2);
    });
    wgpu::Device device;
    auto future = adapter.RequestDevice(&deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestDeviceStatus, wgpu::Device value, wgpu::StringView, wgpu::Device* result) { *result = std::move(value); }, &device);
    instance.WaitAny(future, UINT64_MAX);
    auto shader = readWgslFile(device, SOURCE_DIR "/experiments/ellipsoid.wgsl");
    wgpu::BindGroupLayoutEntry entries[] {
        {.binding = 0, .visibility = wgpu::ShaderStage::Vertex, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
        {.binding = 1, .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, .buffer = {.type = wgpu::BufferBindingType::Uniform}},
    };
    wgpu::BindGroupLayoutDescriptor layoutDesc {.entryCount = 2, .entries = entries};
    auto layout = device.CreateBindGroupLayout(&layoutDesc);
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc {.bindGroupLayoutCount = 1, .bindGroupLayouts = &layout};
    auto pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
    wgpu::ColorTargetState target {.format = colorFormat};
    wgpu::BlendState blend {
        .color = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
    };
    wgpu::FragmentState fragment {.module = shader, .entryPoint = "fs_surface", .targetCount = 1, .targets = &target};
    wgpu::DepthStencilState depth {.format = depthFormat, .depthWriteEnabled = true, .depthCompare = wgpu::CompareFunction::Less};
    wgpu::ConstantEntry constants[] {{.key = "fragment_rays"}, {.key = "normalize_ray"}, {.key = "gaussian_falloff"}};
    wgpu::RenderPipelineDescriptor desc {
        .layout = pipelineLayout,
        .vertex = {.module = shader, .constantCount = 3, .constants = constants},
        .depthStencil = &depth, .fragment = &fragment,
    };
    fragment.constantCount = 3;
    fragment.constants = constants;
    const char* proxies[] {"box", "box_quad", "exact_quad", "golus"};
    const char* vertices[] {"vs_box", "vs_projected_box", "vs_quad", "vs_golus"};
    std::vector<Variant> variants;
    for (int proxy = 0; proxy < 4; proxy++) {
        desc.vertex.entryPoint = vertices[proxy];
        desc.primitive.topology = proxy == 0 ? wgpu::PrimitiveTopology::TriangleList : wgpu::PrimitiveTopology::TriangleStrip;
        for (int stage = 0; stage < 2; stage++) for (int normalized = 0; normalized < 2; normalized++) {
            constants[0].value = stage;
            constants[1].value = normalized;
            std::string name = std::string(proxies[proxy]) + (stage ? "_fragment" : "_vertex") + (normalized ? "_normalized" : "_general");
            variants.push_back({name, device.CreateRenderPipeline(&desc), proxy == 0 ? 36u : 4u});
        }
    }
    desc.vertex.entryPoint = "vs_box";
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::Front;
    for (int stage = 0; stage < 2; stage++) for (int normalized = 0; normalized < 2; normalized++) {
        constants[0].value = stage;
        constants[1].value = normalized;
        std::string name = std::string("box_culled") + (stage ? "_fragment" : "_vertex") + (normalized ? "_normalized" : "_general");
        variants.push_back({name, device.CreateRenderPipeline(&desc), 36});
    }
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
    desc.vertex.entryPoint = "vs_splat";
    fragment.entryPoint = "fs_splat";
    constants[0].value = constants[1].value = 0;
    depth.depthWriteEnabled = false;
    for (int gaussian = 0; gaussian < 2; gaussian++) {
        constants[2].value = gaussian;
        target.blend = gaussian ? &blend : nullptr;
        variants.push_back({gaussian ? "gaussian" : "ellipse", device.CreateRenderPipeline(&desc), 4, true});
    }
    variants.push_back({"empty", {}, 0, true});

    Eigen::Quaternionf q = Eigen::AngleAxisf(0.26f, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(0.61f, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(0.35f, Eigen::Vector3f::UnitX());
    Gaussian shape {{0, 0, 0, 1}, {0.9f, 0.2f, 0.05f, 0}, q.coeffs(), {0.25f, 0.55f, 0.9f, 0.85f}};
    std::vector<Scene> scenes;
    for (std::string name : {"sphere", "ellipsoid", "close", "offaxis"}) {
        Scene scene {name, {shape}, {}};
        scene.camera.distance = name == "close" ? 1.25f : name == "offaxis" ? 2.7f : 5;
        if (name == "sphere") scene.shapes[0].scale = {0.7f, 0.7f, 0.7f, 0};
        if (name == "offaxis") scene.shapes[0].position = {0.9f, 0.4f, 0, 1};
        scenes.push_back(std::move(scene));
    }
    for (bool sphere : {true, false}) {
        Scene scene {sphere ? "grid_spheres" : "grid_ellipsoids", {}, {}};
        scene.camera.distance = 7;
        for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) {
            Gaussian item = shape;
            item.position = {(x - 31.5f) * 0.08f, (y - 31.5f) * 0.08f, 0, 1};
            item.scale = sphere ? Eigen::Vector4f {0.03f, 0.03f, 0.03f, 0} : Eigen::Vector4f {0.035f, 0.008f, 0.002f, 0};
            scene.shapes.push_back(item);
        }
        scenes.push_back(std::move(scene));
    }
    std::mt19937 sceneRandom(10000);
    std::uniform_real_distribution<float> position(-2, 2), logScale(std::log(0.01f), std::log(0.12f));
    std::uniform_real_distribution<float> color(0.2f, 0.95f), opacity(0.1f, 0.9f);
    std::normal_distribution<float> normal;
    std::vector<Gaussian> randomShapes;
    for (int i = 0; i < 10000; i++) {
        Gaussian item;
        item.position = {position(sceneRandom), position(sceneRandom), position(sceneRandom), 1};
        item.scale = {std::exp(logScale(sceneRandom)), std::exp(logScale(sceneRandom)), std::exp(logScale(sceneRandom)), 0};
        item.rotation = Eigen::Vector4f {normal(sceneRandom), normal(sceneRandom), normal(sceneRandom), normal(sceneRandom)}.normalized();
        item.color = {color(sceneRandom), color(sceneRandom), color(sceneRandom), opacity(sceneRandom)};
        randomShapes.push_back(item);
    }
    for (bool close : {false, true}) {
        Scene scene {close ? "random10000_close" : "random10000", randomShapes, {}};
        scene.camera.distance = close ? 3.8f : 7.5f;
        scenes.push_back(std::move(scene));
    }
    auto luigi = readPlyFile(SOURCE_DIR "/assets/luigi.ply");
    Eigen::Vector3f lower = luigi[0].position.head<3>(), upper = lower;
    for (const auto& item : luigi) {
        lower = lower.cwiseMin(item.position.head<3>());
        upper = upper.cwiseMax(item.position.head<3>());
    }
    for (bool close : {false, true}) {
        Scene scene {close ? "luigi_close" : "luigi", luigi, {}};
        scene.camera.center = 0.5f * (lower + upper);
        scene.camera.distance = 0.5f * (upper - lower).maxCoeff() * (close ? 1.75f : 3.5f);
        scenes.push_back(std::move(scene));
    }
    std::ofstream csv(output / "results.csv");
    csv << "scene,width,height,instances,trial,method,gpu_median_ms,gpu_p90_ms,batch_ms,coverage,color_changed,coverage_changed,depth_changed,max_depth_error,samples_ms\n";
    csv << std::fixed << std::setprecision(7);
    std::mt19937 random(20261018);
    auto queue = device.GetQueue();
    for (auto size : {Extent2D {1280, 720}, Extent2D {2560, 1440}}) for (const auto& scene : scenes) {
        if (argc > 2 && !scene.name.starts_with(argv[2])) continue;
        wgpu::BufferDescriptor bufferDesc {.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                                          .size = scene.shapes.size() * sizeof(Gaussian)};
        auto shapes = device.CreateBuffer(&bufferDesc);
        queue.WriteBuffer(shapes, 0, scene.shapes.data(), bufferDesc.size);
        bufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        bufferDesc.size = 144;
        auto camera = device.CreateBuffer(&bufferDesc);
        Eigen::Matrix4f vp = scene.camera.viewProjection(float(size.width) / size.height), inverse = vp.inverse();
        Eigen::Vector4f eye;
        eye << scene.camera.position(), 1;
        queue.WriteBuffer(camera, 0, vp.data(), 64);
        queue.WriteBuffer(camera, 64, eye.data(), 16);
        queue.WriteBuffer(camera, 80, inverse.data(), 64);
        wgpu::BindGroupEntry bindings[] {{.binding = 0, .buffer = shapes, .size = scene.shapes.size() * sizeof(Gaussian)},
                                        {.binding = 1, .buffer = camera, .size = 144}};
        wgpu::BindGroupDescriptor groupDesc {.layout = layout, .entryCount = 2, .entries = bindings};
        auto group = device.CreateBindGroup(&groupDesc);
        for (int trial = 0; trial < 3; trial++) {
            std::vector<int> order(variants.size());
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), random);
            std::vector<Result> results(variants.size());
            for (int index : order) results[index] = measure(instance, device, variants[index], group, size, scene.shapes.size());
            const auto& reference = results[8]; // exact_quad_vertex_general
            for (size_t index = 0; index < variants.size(); index++) {
                auto& result = results[index];
                size_t coverage = 0, colorChanged = 0, coverageChanged = 0, depthChanged = 0;
                float maxDepth = 0;
                for (size_t pixel = 0; pixel < result.depth.size(); pixel++) {
                    bool hit = result.depth[pixel] < 1, refHit = reference.depth[pixel] < 1;
                    coverage += hit;
                    coverageChanged += hit != refHit;
                    int difference = 0;
                    for (int channel = 0; channel < 3; channel++) difference = std::max(difference,
                        std::abs(int(result.color[4 * pixel + channel]) - int(reference.color[4 * pixel + channel])));
                    colorChanged += difference > 5;
                    if (hit && refHit) {
                        float error = std::abs(result.depth[pixel] - reference.depth[pixel]);
                        maxDepth = std::max(maxDepth, error);
                        depthChanged += error > 1e-5f;
                    }
                }
                auto sorted = result.times;
                std::sort(sorted.begin(), sorted.end());
                csv << scene.name << ',' << size.width << ',' << size.height << ',' << scene.shapes.size() << ',' << trial << ','
                    << variants[index].name << ',' << sorted[samples / 2] << ',' << sorted[samples * 9 / 10] << ',' << result.batchMs << ','
                    << coverage << ',' << colorChanged << ',' << coverageChanged << ',' << depthChanged << ',' << maxDepth << ',';
                for (size_t i = 0; i < result.times.size(); i++) csv << (i ? ";" : "") << result.times[i];
                csv << '\n';
                if (trial == 0 && size.width == 1280 && (index % 4 == 0 || variants[index].approximate)) {
                    std::ofstream image(output / (scene.name + "_" + variants[index].name + ".ppm"), std::ios::binary);
                    image << "P6\n" << size.width << ' ' << size.height << "\n255\n";
                    for (size_t i = 0; i < result.color.size(); i += 4) image.write(reinterpret_cast<const char*>(result.color.data() + i), 3);
                }
            }
            csv.flush();
            std::cerr << size.width << 'x' << size.height << ' ' << scene.name << " trial " << trial + 1 << "/3\n";
        }
    }
}

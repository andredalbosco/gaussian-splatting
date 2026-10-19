#include "app.hpp"
#include "camera.hpp"
#include "gaussian.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <imgui.h>

constexpr uint32_t radixWorkgroupSize = 256;
constexpr uint32_t radixBitsPerPass = 4;
constexpr uint32_t radixDigitCount = 1u << radixBitsPerPass;
constexpr uint32_t radixPassCount = 32 / radixBitsPerPass;
constexpr uint32_t radixUniformStride = 256;

struct RadixParams {
    uint32_t count;
    uint32_t groupCount;
    uint32_t blockCount;
    uint32_t bitShift;
};

struct GaussianSplattingApp : App {
    wgpu::RenderPipeline renderPipeline;
    wgpu::ComputePipeline radixHistogramRankPipeline;
    wgpu::ComputePipeline radixScanBlocksPipeline;
    wgpu::ComputePipeline radixScanSumsPipeline;
    wgpu::ComputePipeline radixScatterPipeline;

    wgpu::BindGroup renderBindGroup;
    wgpu::BindGroup radixHistogramBindGroup;
    wgpu::BindGroup radixScanBlocksBindGroup;
    wgpu::BindGroup radixScanSumsBindGroup;
    wgpu::BindGroup radixScatterBindGroup;

    std::array<wgpu::BindGroup, radixPassCount> radixHistogramBindGroups;
    std::array<wgpu::BindGroup, radixPassCount> radixScanBlocksBindGroups;
    std::array<wgpu::BindGroup, radixPassCount> radixScanSumsBindGroups;
    std::array<wgpu::BindGroup, radixPassCount> radixScatterBindGroups;

    wgpu::Buffer gaussianBuffer;
    wgpu::Buffer orderBuffer;
    wgpu::Buffer cameraBuffer;
    wgpu::Buffer radixParamsBuffer;
    wgpu::Buffer radixHistogramBuffer;
    wgpu::Buffer radixBlockOffsetsBuffer;
    wgpu::Buffer radixLocalRanksBuffer;
    std::array<wgpu::Buffer, 2> radixKeysBuffer;

    std::vector<Gaussian> gaussians;
    Eigen::AlignedBox3f bounds;
    Eigen::Vector3f lastSortDirection = Eigen::Vector3f::Zero();
    Eigen::Vector2d mouse = Eigen::Vector2d::Zero();
    OrbitCamera camera;

    uint32_t gaussianCount;
    uint32_t radixGroupCount;
    uint32_t radixBlockCount;
    float scrollDelta = 0;

    bool init();

    bool loadGaussians(const char* path);

    void createPipelines();
    void createBuffers();
    void createBindGroups();

    void recordRadixSort(const wgpu::CommandEncoder& encoder, const wgpu::PassTimestampWrites* timestamps = nullptr) const;

    void update() override;
    void buildUi() override;
    void recordCommands(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target) override;

    void resetCamera();
    void updateCameraBuffer();

    void handleInput();
    void onScroll(double x, double y) override;
};

bool GaussianSplattingApp::init() {
    if (!App::init("Gaussian Splatting")) return false;
    createPipelines();
    return true;
}

bool GaussianSplattingApp::loadGaussians(const char* path) {
    auto newGaussians = readPlyFile(path);
    if (newGaussians.empty()) return false;
    gaussians = std::move(newGaussians);

    gaussianCount = gaussians.size();
    std::cout << "Loaded " << gaussianCount << " Gaussians\n";

    bounds.setEmpty();
    for (const Gaussian& gaussian : gaussians) {
        bounds.extend(gaussian.position.head<3>());
    }
    resetCamera();
    createBuffers();
    createBindGroups();

    return true;
}

void GaussianSplattingApp::createPipelines() {
    wgpu::ShaderModule shader = readWgslFile(device, SRC_DIR "/shaders.wgsl");

    wgpu::BlendState blend {
        .color = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
    };
    wgpu::ColorTargetState target {
        .format = surfaceConfig.format,
        .blend = &blend,
    };

    wgpu::FragmentState fragment {
        .module = shader,
        .entryPoint = "fs_main",
        .targetCount = 1,
        .targets = &target,
    };

    wgpu::RenderPipelineDescriptor pipelineDesc {
        .vertex = {
            .module = shader,
            .entryPoint = "vs_main",
        },
        .primitive = {
            .topology = wgpu::PrimitiveTopology::TriangleStrip,
            .cullMode = wgpu::CullMode::None,
        },
        .fragment = &fragment,
    };
    renderPipeline = device.CreateRenderPipeline(&pipelineDesc);

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

void GaussianSplattingApp::createBuffers() {
    wgpu::BufferDescriptor gaussianBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = gaussians.size() * sizeof(Gaussian),
    };
    gaussianBuffer = device.CreateBuffer(&gaussianBufferDesc);
    queue.WriteBuffer(gaussianBuffer, 0, gaussians.data(), gaussianBufferDesc.size);

    wgpu::BufferDescriptor orderBufferDesc {
        .usage = wgpu::BufferUsage::Storage,
        .size = gaussianCount * sizeof(uint32_t),
    };
    orderBuffer = device.CreateBuffer(&orderBufferDesc);

    wgpu::BufferDescriptor cameraBufferDesc {
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = 2 * sizeof(Eigen::Matrix4f) + sizeof(Eigen::Vector4f),
    };
    cameraBuffer = device.CreateBuffer(&cameraBufferDesc);

    radixGroupCount = (gaussianCount + radixWorkgroupSize - 1) / radixWorkgroupSize;
    radixBlockCount = (radixGroupCount + radixWorkgroupSize - 1) / radixWorkgroupSize;

    wgpu::BufferDescriptor keysBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        .size = std::max<uint64_t>(16, uint64_t(gaussianCount) * 2 * sizeof(uint32_t)),
    };
    radixKeysBuffer[0] = device.CreateBuffer(&keysBufferDesc);
    radixKeysBuffer[1] = device.CreateBuffer(&keysBufferDesc);

    wgpu::BufferDescriptor localRanksBufferDesc {
        .usage = wgpu::BufferUsage::Storage,
        .size = std::max<uint64_t>(16, uint64_t(gaussianCount) * sizeof(uint32_t)),
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
            .count = gaussianCount,
            .groupCount = radixGroupCount,
            .blockCount = radixBlockCount,
            .bitShift = passIndex * radixBitsPerPass,
        };
        device.GetQueue().WriteBuffer(radixParamsBuffer, passIndex * radixUniformStride, &radixParams, sizeof(radixParams));
    }
    lastSortDirection.setZero();
}

void GaussianSplattingApp::createBindGroups() {
    wgpu::BindGroupEntry entries[] {
        {
            .binding = 0,
            .buffer = gaussianBuffer,
            .size = gaussianBuffer.GetSize(),
        },
        {
            .binding = 1,
            .buffer = cameraBuffer,
            .size = cameraBuffer.GetSize(),
        },
        {
            .binding = 2,
            .buffer = orderBuffer,
            .size = orderBuffer.GetSize(),
        },
    };

    wgpu::BindGroupDescriptor bindGroupDesc {
        .layout = renderPipeline.GetBindGroupLayout(0),
        .entryCount = 3,
        .entries = entries,
    };
    renderBindGroup = device.CreateBindGroup(&bindGroupDesc);

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

void GaussianSplattingApp::update() {
    handleInput();
    updateCameraBuffer();
}

void GaussianSplattingApp::recordCommands(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target) {
    Eigen::Vector3f direction = (camera.center - camera.position()).normalized();
    if ((direction - lastSortDirection).squaredNorm() != 0) {
        recordRadixSort(encoder);
        lastSortDirection = direction;
    }
    wgpu::RenderPassColorAttachment color {
        .view = target,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.05, 0.05, 0.07, 1.0},
    };
    wgpu::RenderPassDescriptor passDesc {
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
    };
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(renderPipeline);
    pass.SetBindGroup(0, renderBindGroup);
    pass.Draw(4, gaussianCount);
    pass.End();
}

void GaussianSplattingApp::buildUi() {
    float uiScale = ImGui::GetStyle().FontScaleDpi;
    ImGui::SetNextWindowPos({12 * uiScale, 12 * uiScale}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({340 * uiScale, 480 * uiScale}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Gaussian Splatting");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    ImGui::Text("%u Gaussians", gaussianCount);
    ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);

    ImGui::SeparatorText("Projection");
    ImGui::SliderFloat("FOV (degrees)", &camera.fovy, 15, 120, "%.1f");
    ImGui::DragFloat("Near", &camera.zNear, 0.001f, 0.0001f, camera.zFar - 0.001f, "%.4f");
    ImGui::DragFloat("Far", &camera.zFar, 0.1f, camera.zNear + 0.001f, 1000.0f, "%.1f");

    ImGui::SeparatorText("Orbit");
    ImGui::SliderFloat("Sensitivity", &camera.orbitSensitivity, 0.01f, 1.0f, "%.4f");
    ImGui::SliderFloat("Yaw (degrees)", &camera.yaw, -360, 360, "%.1f");
    ImGui::SliderFloat("Pitch limit (degrees)", &camera.maxPitch, 10, 89, "%.1f");
    ImGui::SliderFloat("Pitch (degrees)", &camera.pitch, -camera.maxPitch, camera.maxPitch, "%.2f");

    ImGui::SeparatorText("Zoom");
    ImGui::SliderFloat("Zoom sensitivity", &camera.zoomSensitivity, 0.001f, 0.5f, "%.3f");
    ImGui::DragFloat("Min distance", &camera.minDistance, 0.01f, 0.001f, camera.maxDistance);
    ImGui::DragFloat("Max distance", &camera.maxDistance, 0.1f, camera.minDistance, 1000.0f);
    ImGui::SliderFloat(
        "Distance",
        &camera.distance,
        camera.minDistance,
        camera.maxDistance,
        "%.3f",
        ImGuiSliderFlags_Logarithmic
    );
    if (ImGui::Button("Reset camera")) resetCamera();
    ImGui::PopItemWidth();
    ImGui::End();
}

void GaussianSplattingApp::resetCamera() {
    float radius = 0.5f * bounds.sizes().maxCoeff();
    camera.center = bounds.center();
    camera.distance = 3.5f * radius;
    camera.minDistance = 0.3f * radius;
    camera.maxDistance = 20.0f * radius;
    camera.yaw = 10;
    camera.pitch = 5;
}

void GaussianSplattingApp::updateCameraBuffer() {
    Eigen::Matrix4f view = viewMatrix(camera.position(), camera.center, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f projection = projectionMatrix(
        camera.fovy, float(surfaceConfig.width) / surfaceConfig.height, camera.zNear, camera.zFar);
    Eigen::Matrix4f viewProjection = projection * view;
    Eigen::Vector4f viewport {
        float(surfaceConfig.width), float(surfaceConfig.height),
        projection(0, 0), projection(1, 1),
    };
    queue.WriteBuffer(cameraBuffer, 0, viewProjection.data(), sizeof(viewProjection));
    queue.WriteBuffer(cameraBuffer, sizeof(viewProjection), view.data(), sizeof(view));
    queue.WriteBuffer(cameraBuffer, sizeof(viewProjection) + sizeof(view), viewport.data(), sizeof(viewport));
}

void GaussianSplattingApp::handleInput() {
    Eigen::Vector2d position;
    glfwGetCursorPos(window, &position.x(), &position.y());

    if (!ImGui::GetIO().WantCaptureMouse) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            camera.orbit((position - mouse).cast<float>());
        }
        camera.zoom(scrollDelta);
    }

    scrollDelta = 0;
    mouse = position;
}

void GaussianSplattingApp::onScroll(double, double y) {
    scrollDelta += static_cast<float>(y);
}

void GaussianSplattingApp::recordRadixSort(const wgpu::CommandEncoder& encoder, const wgpu::PassTimestampWrites* timestamps) const {
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

int main(int argc, char* argv[]) {
    static GaussianSplattingApp app {};
    if (!app.init()) return 1;
    if (!app.loadGaussians(SRC_DIR "/../assets/luigi.ply")) {
        app.shutdown();
        return 1;
    }

    if (argc > 1 && std::string_view(argv[1]) == "--capture") {
        app.frame();
        app.shutdown();
    } else {
        app.run();
    }
    return 0;
}

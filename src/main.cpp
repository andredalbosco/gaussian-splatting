#include "app.hpp"
#include "camera.hpp"
#include "gaussian.hpp"
#include "radix_sort.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>
#include <utility>

#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <imgui.h>

constexpr const char* assetPaths[] {
    ASSETS_DIR "/black_jacket.ply",
    ASSETS_DIR "/luigi.ply",
};

struct GaussianSplattingApp : App {
    RadixSort radixSort;

    wgpu::RenderPipeline renderPipeline;
    wgpu::BindGroup renderBindGroup;

    wgpu::Buffer gaussianBuffer;
    wgpu::Buffer cameraBuffer;
    wgpu::Buffer orderBuffer;

    uint32_t gaussianCount;
    int assetIndex = 0;

    std::vector<Gaussian> gaussians;
    Eigen::AlignedBox3f bounds;
    Eigen::Vector3f lastSortDirection = Eigen::Vector3f::Zero();
    Eigen::Vector2d mouse = Eigen::Vector2d::Zero();
    OrbitCamera camera;
    Eigen::Vector3f cameraUp = Eigen::Vector3f::UnitY();
    float scrollDelta = 0;

    bool init();
    bool loadGaussians(const char* path);
    void createPipelines(const wgpu::ShaderModule& shader, wgpu::TextureFormat format);
    void createBuffers();
    void createBindGroups();
    void updateCameraBuffer();

    void update() override;
    void buildUi() override;
    void recordCommands(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target) override;

    void resetCamera();
    void handleInput();
    void onScroll(double x, double y) override;
};

bool GaussianSplattingApp::init() {
    if (!App::init("Gaussian Splatting")) return false;
    auto shader = readWgslFile(device, SRC_DIR "/shaders.wgsl");
    createPipelines(shader, surfaceConfig.format);
    return true;
}

bool GaussianSplattingApp::loadGaussians(const char* path) {
    auto newGaussians = readPlyFile(path);
    if (newGaussians.empty()) return false;
    gaussians = std::move(newGaussians);

    std::cout << "Loaded " << gaussians.size() << " Gaussians\n";

    bounds.setEmpty();
    for (const Gaussian& gaussian : gaussians) {
        bounds.extend(gaussian.position.head<3>());
    }
    resetCamera();
    createBuffers();
    createBindGroups();

    return true;
}

void GaussianSplattingApp::createPipelines(const wgpu::ShaderModule& shader, wgpu::TextureFormat format) {
    wgpu::BlendState blend {
        .color = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
    };
    wgpu::ColorTargetState target {
        .format = format,
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
    radixSort.init(device, shader);
}

void GaussianSplattingApp::createBuffers() {
    gaussianCount = gaussians.size();
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
    radixSort.createBuffers(gaussianCount);
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
    radixSort.createBindGroups(gaussianBuffer, cameraBuffer, orderBuffer);
}

void GaussianSplattingApp::updateCameraBuffer() {
    Eigen::Matrix4f view = viewMatrix(camera.position(), camera.center, cameraUp);
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

void GaussianSplattingApp::update() {
    handleInput();
    updateCameraBuffer();
}

void GaussianSplattingApp::recordCommands(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target) {
    Eigen::Vector3f direction = (camera.center - camera.position()).normalized();
    if ((direction - lastSortDirection).squaredNorm() != 0) {
        radixSort.recordCommands(encoder);
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
    ImGui::Begin("Gaussian Splatting", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    if (ImGui::Combo("Asset", &assetIndex, "Black Jacket\0Luigi\0")) {
        loadGaussians(assetPaths[assetIndex]);
    }
    ImGui::Text("%u Gaussians", gaussianCount);
    ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);

    ImGui::SeparatorText("Projection");
    ImGui::SliderFloat("FOV (degrees)", &camera.fovy, 15, 120, "%.1f");
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
    cameraUp = {0, assetIndex == 0 ? -1.0f : 1.0f, 0};
    camera.pitch = 5 * cameraUp.y();
}

void GaussianSplattingApp::handleInput() {
    Eigen::Vector2d position;
    glfwGetCursorPos(window, &position.x(), &position.y());

    if (!ImGui::GetIO().WantCaptureMouse) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            camera.orbit((position - mouse).cast<float>() * cameraUp.y());
        }
        camera.zoom(scrollDelta);
    }

    scrollDelta = 0;
    mouse = position;
}

void GaussianSplattingApp::onScroll(double, double y) {
    scrollDelta += static_cast<float>(y);
}

int main(int argc, char* argv[]) {
    static GaussianSplattingApp app {};
    if (!app.init()) return 1;
    if (!app.loadGaussians(assetPaths[app.assetIndex])) {
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

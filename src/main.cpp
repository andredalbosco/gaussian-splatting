#include "camera.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

struct GSplatApp {
    GLFWwindow* window;
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;

    wgpu::Surface surface;
    wgpu::SurfaceConfiguration surfaceConfig;

    wgpu::RenderPipeline pipeline;
    wgpu::BindGroup bindGroup;

    wgpu::Buffer gaussianBuffer;
    wgpu::Buffer orderBuffer;
    wgpu::Buffer cameraBuffer;

    std::vector<Gaussian> gaussians;
    uint32_t gaussianCount;
    Eigen::AlignedBox3f bounds;

    std::vector<uint32_t> order;
    std::vector<float> depths;
    Eigen::Vector3f lastSortDirection = Eigen::Vector3f::Zero();

    OrbitCamera camera;

    Eigen::Vector2d mouse = Eigen::Vector2d::Zero();
    float scrollDelta = 0;

    bool init();
    void run();
    void shutdown();

    bool loadGaussians(const char* path);

    void createPipeline();
    void createBuffers();
    void createBindGroup();

    void drawFrame();
    void drawUi();

    void resetCamera();
    void updateCamera();

    void handleInput();
    static void onScroll(GLFWwindow* window, double x, double y);
};

bool GSplatApp::init() {
    initGlfw(800, 600, "Gaussian Splatting", window);
    initWgpu(window, instance, device, surface, surfaceConfig);
    if (!device) return false;

    queue = device.GetQueue();
    glfwSetWindowUserPointer(window, this);
    glfwSetScrollCallback(window, onScroll);

    createPipeline();

    initImgui(device, window, surfaceConfig.format);
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    return true;
}

void GSplatApp::run() {
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg([](void* arg) {
        static_cast<GSplatApp*>(arg)->drawFrame();
    }, this, 0, false);
#else
    while (!glfwWindowShouldClose(window)) {
        drawFrame();
    }

    shutdown();
#endif
}

void GSplatApp::shutdown() {
    shutdownImgui();
    shutdownWgpu(surface);
    shutdownGlfw(window);
}

bool GSplatApp::loadGaussians(const char* path) {
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
    lastSortDirection.setZero();
    createBuffers();
    createBindGroup();

    return true;
}

void GSplatApp::createPipeline() {
    wgpu::ShaderModule shader = readWgslFile(device, SOURCE_DIR "/src/shaders.wgsl");

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
    pipeline = device.CreateRenderPipeline(&pipelineDesc);
}

void GSplatApp::createBuffers() {
    wgpu::BufferDescriptor gaussianBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = gaussians.size() * sizeof(Gaussian),
    };
    gaussianBuffer = device.CreateBuffer(&gaussianBufferDesc);
    queue.WriteBuffer(gaussianBuffer, 0, gaussians.data(), gaussianBufferDesc.size);

    order.resize(gaussianCount);
    depths.resize(gaussianCount);
    std::iota(order.begin(), order.end(), 0u);

    wgpu::BufferDescriptor orderBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = gaussianCount * sizeof(uint32_t),
    };
    orderBuffer = device.CreateBuffer(&orderBufferDesc);

    wgpu::BufferDescriptor cameraBufferDesc {
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Eigen::Matrix4f) + sizeof(Eigen::Vector4f),
    };
    cameraBuffer = device.CreateBuffer(&cameraBufferDesc);
}

void GSplatApp::createBindGroup() {
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
        .layout = pipeline.GetBindGroupLayout(0),
        .entryCount = 3,
        .entries = entries,
    };
    bindGroup = device.CreateBindGroup(&bindGroupDesc);
}

void GSplatApp::drawFrame() {
    glfwPollEvents();
    if (!handleWindowResize(surface, window, surfaceConfig)) return;

    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    drawUi();
    handleInput();
    updateCamera();
    ImGui::Render();

    wgpu::SurfaceTexture frame;
    surface.GetCurrentTexture(&frame);
    if (!frame.texture) return;

    wgpu::RenderPassColorAttachment color {
        .view = frame.texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.05, 0.05, 0.07, 1.0},
    };

    wgpu::RenderPassDescriptor passDesc {
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
    };

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(4, gaussianCount);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass.Get());
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

#ifndef __EMSCRIPTEN__
    surface.Present();
#endif
    instance.ProcessEvents();
}

void GSplatApp::drawUi() {
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

void GSplatApp::resetCamera() {
    float radius = 0.5f * bounds.sizes().maxCoeff();
    camera.center = bounds.center();
    camera.distance = 3.5f * radius;
    camera.minDistance = 0.3f * radius;
    camera.maxDistance = 20.0f * radius;
    camera.yaw = 10;
    camera.pitch = 5;
}

void GSplatApp::updateCamera() {
    Eigen::Matrix4f viewProjection = camera.viewProjection(
        float(surfaceConfig.width) / surfaceConfig.height);
    queue.WriteBuffer(cameraBuffer, 0, viewProjection.data(), sizeof(viewProjection));
    Eigen::Vector4f viewport {float(surfaceConfig.width), float(surfaceConfig.height), 0, 0};
    queue.WriteBuffer(cameraBuffer, sizeof(viewProjection), viewport.data(), sizeof(viewport));

    Eigen::Vector3f direction = (camera.center - camera.position()).normalized();
    if ((direction - lastSortDirection).squaredNorm() == 0) return;
    lastSortDirection = direction;
    for (uint32_t i = 0; i < gaussianCount; i++)
        depths[i] = direction.dot(gaussians[i].position.head<3>());
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return depths[a] == depths[b] ? a < b : depths[a] > depths[b];
    });
    queue.WriteBuffer(orderBuffer, 0, order.data(), order.size() * sizeof(uint32_t));
}

void GSplatApp::handleInput() {
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

void GSplatApp::onScroll(GLFWwindow* window, double, double y) {
    GSplatApp* app = static_cast<GSplatApp*>(glfwGetWindowUserPointer(window));
    app->scrollDelta += static_cast<float>(y);
}

int main(int argc, char* argv[]) {
    static GSplatApp app {};
    if (!app.init()) return 1;
    if (!app.loadGaussians(SOURCE_DIR "/assets/luigi.ply")) {
        app.shutdown();
        return 1;
    }

    if (argc > 1 && std::string_view(argv[1]) == "--capture") {
        app.drawFrame();
        app.shutdown();
    } else {
        app.run();
    }
    return 0;
}

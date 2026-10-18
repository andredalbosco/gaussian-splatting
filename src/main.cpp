#include "camera.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

#include <iostream>
#include <string_view>
#include <vector>

#include <Eigen/Core>
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
    wgpu::TextureFormat colorFormat;
    static constexpr wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Depth24Plus;

    wgpu::RenderPipeline pipeline;
    wgpu::BindGroup bindGroup;
    wgpu::Texture depthTexture;

    wgpu::Buffer gaussianBuffer;
    uint32_t gaussianCount;

    OrbitCamera camera;
    float initialCameraDistance;
    static constexpr float fitDistanceFactor = 3.5f;
    static constexpr float minDistanceFactor = 0.3f;
    static constexpr float maxDistanceFactor = 20.0f;
    wgpu::Buffer cameraBuffer;

    Eigen::Vector2d mouse = Eigen::Vector2d::Zero();
    float scrollDelta = 0;

    bool init();
    void shutdown();
    void run();

    bool loadScene();

    void initPipeline();
    void drawFrame();
    void drawUi();

    void updateCamera();

    void handleInput();
    static void onScroll(GLFWwindow* window, double x, double y);
};

bool GSplatApp::init() {
    initGlfw(800, 600, "Gaussian Splatting - Luigi", window);
    initWgpu(window, instance, device, surface, surfaceConfig);
    if (!device) return false;

    colorFormat = surfaceConfig.format;
    queue = device.GetQueue();
    glfwSetWindowUserPointer(window, this);
    glfwSetScrollCallback(window, onScroll);

    if (!loadScene()) return false;
    initPipeline();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOther(window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
    ImGui_ImplWGPU_InitInfo imguiInfo;
    imguiInfo.Device = device.Get();
    imguiInfo.RenderTargetFormat = static_cast<WGPUTextureFormat>(colorFormat);
    imguiInfo.DepthStencilFormat = static_cast<WGPUTextureFormat>(depthFormat);
    ImGui_ImplWGPU_Init(&imguiInfo);

    return true;
}

void GSplatApp::shutdown() {
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    shutdownWgpu(surface);
    shutdownGlfw(window);
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

bool GSplatApp::loadScene() {
    std::vector<Gaussian> gaussians = readPlyFile(SOURCE_DIR "/assets/luigi.ply");
    if (gaussians.empty()) return false;

    gaussianCount = gaussians.size();
    std::cout << "Loaded " << gaussianCount << " Gaussians\n";

    Eigen::Vector3f lower = gaussians[0].position.head<3>();
    Eigen::Vector3f upper = lower;

    for (const Gaussian& gaussian : gaussians) {
        lower = lower.cwiseMin(gaussian.position.head<3>());
        upper = upper.cwiseMax(gaussian.position.head<3>());
    }

    float radius = 0.5f * (upper - lower).maxCoeff();
    camera.center = 0.5f * (lower + upper);
    camera.distance = fitDistanceFactor * radius;
    camera.minDistance = minDistanceFactor * radius;
    camera.maxDistance = maxDistanceFactor * radius;
    initialCameraDistance = camera.distance;

    wgpu::BufferDescriptor gaussianBufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = gaussians.size() * sizeof(Gaussian),
    };
    gaussianBuffer = device.CreateBuffer(&gaussianBufferDesc);
    queue.WriteBuffer(gaussianBuffer, 0, gaussians.data(), gaussianBufferDesc.size);

    Eigen::Matrix4f viewProjection = Eigen::Matrix4f::Identity();
    wgpu::BufferDescriptor cameraBufferDesc {
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(viewProjection) + sizeof(Eigen::Vector4f),
    };
    cameraBuffer = device.CreateBuffer(&cameraBufferDesc);
    queue.WriteBuffer(cameraBuffer, 0, viewProjection.data(), sizeof(viewProjection));

    return true;
}

void GSplatApp::initPipeline() {
    wgpu::ShaderModule shader = readWgslFile(device, SOURCE_DIR "/src/shaders.wgsl");

    wgpu::ColorTargetState target {
        .format = colorFormat,
    };

    wgpu::FragmentState fragment {
        .module = shader,
        .entryPoint = "fs_main",
        .targetCount = 1,
        .targets = &target,
    };

    wgpu::DepthStencilState depthState {
        .format = depthFormat,
        .depthWriteEnabled = true,
        .depthCompare = wgpu::CompareFunction::Less,
    };

    wgpu::RenderPipelineDescriptor pipelineDesc {
        .vertex = {
            .module = shader,
            .entryPoint = "vs_main",
        },
        .primitive = {
            .topology = wgpu::PrimitiveTopology::TriangleList,
            .cullMode = wgpu::CullMode::Front,
        },
        .depthStencil = &depthState,
        .fragment = &fragment,
    };
    pipeline = device.CreateRenderPipeline(&pipelineDesc);

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
    };

    wgpu::BindGroupDescriptor bindGroupDesc {
        .layout = pipeline.GetBindGroupLayout(0),
        .entryCount = 2,
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

    if (!depthTexture || depthTexture.GetWidth() != surfaceConfig.width ||
        depthTexture.GetHeight() != surfaceConfig.height) {
        if (depthTexture) depthTexture.Destroy();

        wgpu::TextureDescriptor depthDesc {
            .usage = wgpu::TextureUsage::RenderAttachment,
            .size = {surfaceConfig.width, surfaceConfig.height, 1},
            .format = depthFormat,
        };
        depthTexture = device.CreateTexture(&depthDesc);
    }

    wgpu::SurfaceTexture frame;
    surface.GetCurrentTexture(&frame);
    if (!frame.texture) return;

    wgpu::RenderPassColorAttachment color {
        .view = frame.texture.CreateView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = {0.05, 0.05, 0.07, 1.0},
    };

    wgpu::RenderPassDepthStencilAttachment depth {
        .view = depthTexture.CreateView(),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 1.0f,
    };

    wgpu::RenderPassDescriptor passDesc {
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
        .depthStencilAttachment = &depth,
    };

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bindGroup);
    pass.Draw(36, gaussianCount);
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
    ImGui::SetNextWindowPos({12, 12}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({340, 480}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Gaussian Splatting");
    ImGui::PushItemWidth(150);
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
    if (ImGui::Button("Reset camera")) {
        camera.distance = initialCameraDistance;
        camera.yaw = 10;
        camera.pitch = 5;
    }
    ImGui::PopItemWidth();
    ImGui::End();
}

void GSplatApp::updateCamera() {
    Eigen::Matrix4f viewProjection = camera.viewProjection(
        float(surfaceConfig.width) / surfaceConfig.height);
    queue.WriteBuffer(cameraBuffer, 0, viewProjection.data(), sizeof(viewProjection));
    Eigen::Vector4f position;
    position << camera.position(), 1.0f;
    queue.WriteBuffer(cameraBuffer, sizeof(viewProjection), position.data(), sizeof(position));
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

void GSplatApp::onScroll(GLFWwindow* window, double x, double y) {
    GSplatApp* app = static_cast<GSplatApp*>(glfwGetWindowUserPointer(window));
    app->scrollDelta += static_cast<float>(y);
}

int main(int argc, char* argv[]) {
    static GSplatApp app {};
    if (!app.init()) return 1;

    if (argc > 1 && std::string_view(argv[1]) == "--capture") {
        app.drawFrame();
        app.shutdown();
        return 0;
    }

    app.run();
}

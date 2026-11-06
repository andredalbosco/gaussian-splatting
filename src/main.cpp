#include "camera.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

#ifndef __EMSCRIPTEN__
#include "benchmark.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
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

    wgpu::RenderPipeline pipelines[3];
    wgpu::RenderPipeline quadWirePipelines[3];
    wgpu::RenderPipeline boxWirePipeline;
    wgpu::BindGroup bindGroup;
    wgpu::Texture depthTexture;

    wgpu::Buffer gaussianBuffer;
    uint32_t gaussianCount;
    int scene = 0;
    int method = 0;
    bool showEllipsoid = true;
    bool showQuads = false;
    bool showBoxes = false;
    Eigen::Vector3f testScale {0.9f, 0.5f, 0.3f};
    Eigen::Vector3f testRotation {20, 35, 15};

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
    Gaussian testEllipsoid() const;

    void initPipeline();
    void drawFrame();
    void drawUi();

    void updateCamera();
#ifndef __EMSCRIPTEN__
    void benchmark();
#endif

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

Gaussian GSplatApp::testEllipsoid() const {
    Eigen::Vector3f radians = testRotation * std::numbers::pi_v<float> / 180.0f;
    Eigen::Quaternionf rotation = Eigen::AngleAxisf(radians.z(), Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(radians.y(), Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(radians.x(), Eigen::Vector3f::UnitX());
    return {
        .position = {0, 0, 0, 0},
        .scale = {testScale.x(), testScale.y(), testScale.z(), 0},
        .rotation = rotation.coeffs(),
        .color = {0.25f, 0.55f, 0.9f, 1},
    };
}

bool GSplatApp::loadScene() {
    glfwSetWindowTitle(window, scene == 0 ? "Gaussian Splatting - Luigi" : "Gaussian Splatting - Test ellipsoid");
    std::vector<Gaussian> gaussians;
    if (scene == 0) gaussians = readPlyFile(SOURCE_DIR "/assets/luigi.ply");
    else gaussians.push_back(testEllipsoid());
    if (gaussians.empty()) return false;

    gaussianCount = gaussians.size();
    std::cout << "Loaded " << gaussianCount << " Gaussians\n";

    Eigen::Vector3f lower = gaussians[0].position.head<3>();
    Eigen::Vector3f upper = lower;

    for (const Gaussian& gaussian : gaussians) {
        lower = lower.cwiseMin(gaussian.position.head<3>());
        upper = upper.cwiseMax(gaussian.position.head<3>());
    }

    float radius = scene == 0 ? 0.5f * (upper - lower).maxCoeff() : testScale.maxCoeff();
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
        .size = 2 * sizeof(viewProjection) + 2 * sizeof(Eigen::Vector4f),
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

    wgpu::BindGroupLayoutEntry layoutEntries[] {
        {
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage},
        },
        {
            .binding = 1,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::Uniform},
        },
    };
    wgpu::BindGroupLayoutDescriptor layoutDesc {
        .entryCount = 2,
        .entries = layoutEntries,
    };
    wgpu::BindGroupLayout layout = device.CreateBindGroupLayout(&layoutDesc);
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &layout,
    };

    wgpu::RenderPipelineDescriptor pipelineDesc {
        .layout = device.CreatePipelineLayout(&pipelineLayoutDesc),
        .vertex = {
            .module = shader,
            .entryPoint = "vs_main",
        },
        .primitive = {
            .topology = wgpu::PrimitiveTopology::TriangleStrip,
            .cullMode = wgpu::CullMode::None,
        },
        .depthStencil = &depthState,
        .fragment = &fragment,
    };
    wgpu::ConstantEntry methodConstants[] {{.key = "golus_billboard"}, {.key = "object_space_rays"}};
    pipelineDesc.vertex.constantCount = 2;
    pipelineDesc.vertex.constants = methodConstants;
    for (int i = 0; i < 3; i++) {
        methodConstants[0].value = i == 1;
        methodConstants[1].value = i != 0;
        depthState.depthWriteEnabled = true;
        depthState.depthCompare = wgpu::CompareFunction::Less;
        pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
        pipelineDesc.vertex.entryPoint = "vs_main";
        fragment.entryPoint = "fs_main";
        fragment.constantCount = 2;
        fragment.constants = methodConstants;
        pipelines[i] = device.CreateRenderPipeline(&pipelineDesc);

        pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::LineList;
        depthState.depthWriteEnabled = false;
        depthState.depthCompare = wgpu::CompareFunction::Always;
        pipelineDesc.vertex.entryPoint = "vs_quad_wire";
        fragment.entryPoint = "fs_wire";
        fragment.constantCount = 0;
        fragment.constants = nullptr;
        quadWirePipelines[i] = device.CreateRenderPipeline(&pipelineDesc);
    }
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;
    pipelineDesc.vertex.entryPoint = "vs_box_wire";
    boxWirePipeline = device.CreateRenderPipeline(&pipelineDesc);

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
        .layout = layout,
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
    pass.SetBindGroup(0, bindGroup);
    if (showEllipsoid) {
        pass.SetPipeline(pipelines[method]);
        pass.Draw(4, gaussianCount);
    }
    if (showBoxes) {
        pass.SetPipeline(boxWirePipeline);
        pass.Draw(24, gaussianCount);
    }
    if (showQuads) {
        pass.SetPipeline(quadWirePipelines[method]);
        pass.Draw(10, gaussianCount);
    }
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
    ImGui::SetNextWindowSize({340, 680}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Gaussian Splatting");
    ImGui::PushItemWidth(150);
    if (ImGui::Combo("Scene", &scene, "Luigi\0Test ellipsoid\0")) {
        showEllipsoid = true;
        showQuads = scene == 1;
        showBoxes = scene == 1;
        loadScene();
        initPipeline();
    }
    ImGui::Text("%u Gaussians", gaussianCount);
    ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);

    ImGui::SeparatorText("Rendering");
    ImGui::Combo("Method", &method, "Exact ellipsoid bounds\0Golus billboard\0Exact + object rays\0");
    ImGui::Checkbox("Ellipsoid", &showEllipsoid);
    ImGui::Checkbox("Quad wireframe (cyan)", &showQuads);
    ImGui::Checkbox("Box wireframe (orange)", &showBoxes);
    if (scene == 1) {
        ImGui::SeparatorText("Test ellipsoid");
        bool changed = ImGui::DragFloat3("Scale", testScale.data(), 0.01f, 0.05f, 2.0f, "%.2f");
        changed |= ImGui::DragFloat3("Rotation (degrees)", testRotation.data(), 1, -180, 180, "%.0f");
        if (ImGui::Button("Sphere")) {
            testScale = Eigen::Vector3f::Constant(0.7f);
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Ellipsoid preset")) {
            testScale = {0.9f, 0.5f, 0.3f};
            testRotation = {20, 35, 15};
            changed = true;
        }
        if (changed) {
            Gaussian gaussian = testEllipsoid();
            queue.WriteBuffer(gaussianBuffer, 0, &gaussian, sizeof(gaussian));
        }
    }

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
    Eigen::Matrix4f inverseViewProjection = viewProjection.inverse();
    queue.WriteBuffer(cameraBuffer, sizeof(viewProjection) + sizeof(position),
                      inverseViewProjection.data(), sizeof(inverseViewProjection));
    uint32_t settings[4] {static_cast<uint32_t>(scene), 0, 0, 0};
    queue.WriteBuffer(cameraBuffer, 2 * sizeof(viewProjection) + sizeof(position), settings, sizeof(settings));
}

#ifndef __EMSCRIPTEN__
void GSplatApp::benchmark() {
    if (!device.HasFeature(wgpu::FeatureName::TimestampQuery)) {
        std::cerr << "Timestamp queries are unavailable\n";
        return;
    }
    wgpu::AdapterInfo info;
    device.GetAdapter().GetInfo(&info);
    std::cout << "GPU: " << std::string_view(info.device.data, info.device.length) << '\n';
    std::cout << "case,width,height,method,gpu_median_ms,gpu_p90_ms,throughput_ms,changed_pixels\n";
    std::cout << std::fixed << std::setprecision(4);
    const char* cases[] {"Luigi framed", "Luigi close", "Luigi minimum distance", "Test ellipsoid", "Test sphere"};
    for (Extent2D size : {Extent2D {1280, 720}, Extent2D {2560, 1440}}) {
        surfaceConfig.width = size.width;
        surfaceConfig.height = size.height;
        for (int test = 0; test < 5; test++) {
            scene = test < 3 ? 0 : 1;
            testScale = test == 4 ? Eigen::Vector3f::Constant(0.7f) : Eigen::Vector3f {0.9f, 0.5f, 0.3f};
            loadScene();
            initPipeline();
            camera.yaw = 10;
            camera.pitch = 5;
            if (test == 1) camera.distance *= 0.5f;
            if (test == 2) camera.distance = camera.minDistance;
            updateCamera();
            std::vector<uint8_t> reference;
            const char* methods[] {"Exact", "Golus", "Exact object rays"};
            for (int variant = 0; variant < 3; variant++) {
                BenchmarkResult result = measureRendering(instance, device, pipelines[variant], bindGroup,
                                                          colorFormat, size, gaussianCount);
                size_t changed = 0;
                if (variant == 0) reference = result.pixels;
                else {
                    for (size_t pixel = 0; pixel < reference.size(); pixel += 4) {
                        int difference = 0;
                        for (int channel = 0; channel < 3; channel++) {
                            difference = std::max(difference, std::abs(int(reference[pixel + channel]) -
                                                                      int(result.pixels[pixel + channel])));
                        }
                        changed += difference > 5;
                    }
                }
                std::cout << cases[test] << ',' << size.width << ',' << size.height << ','
                          << methods[variant] << ',' << result.gpuMedianMs << ','
                          << result.gpuP90Ms << ',' << result.throughputMs << ',' << changed << '\n';
            }
        }
    }
}
#endif

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

#ifndef __EMSCRIPTEN__
    if (argc > 1 && std::string_view(argv[1]) == "--benchmark") {
        app.benchmark();
        app.shutdown();
        return 0;
    }
#endif

    if (argc > 1 && std::string_view(argv[1]) == "--capture") {
        app.drawFrame();
        app.shutdown();
        return 0;
    }

    app.run();
}

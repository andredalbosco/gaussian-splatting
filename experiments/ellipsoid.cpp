#include "camera.hpp"
#include "gaussian.hpp"
#include "glfw_utils.hpp"
#include "imgui_utils.hpp"
#include "wgpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string_view>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

struct EllipsoidExperiment {
    GLFWwindow* window;
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Surface surface;
    wgpu::SurfaceConfiguration config;
    wgpu::RenderPipeline pipelines[4], wires[2];
    wgpu::Buffer shapeBuffer, cameraBuffers[4];
    wgpu::BindGroup groups[4];
    wgpu::Texture depthTexture;
    OrbitCamera camera;
    Eigen::Vector3f scale {0.9f, 0.5f, 0.3f};
    Eigen::Vector3f rotation {20, 35, 15};
    Eigen::Vector2d mouse = Eigen::Vector2d::Zero();
    float scroll = 0;
    float opacity = 0.85f;
    bool showBounds = true;
    static constexpr auto depthFormat = wgpu::TextureFormat::Depth24Plus;

    void init();
    void draw();
    void shutdown();
};

void EllipsoidExperiment::init() {
    initGlfw(1200, 900, "Ellipsoid experiments", window);
    initWgpu(window, instance, device, surface, config);
    camera.distance = 5;
    camera.minDistance = 1.8f;
    camera.maxDistance = 20;
    glfwSetWindowUserPointer(window, this);
    glfwSetScrollCallback(window, [](GLFWwindow* window, double, double y) {
        static_cast<EllipsoidExperiment*>(glfwGetWindowUserPointer(window))->scroll += y;
    });
    glfwGetCursorPos(window, &mouse.x(), &mouse.y());

    wgpu::BufferDescriptor bufferDesc {
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Gaussian),
    };
    shapeBuffer = device.CreateBuffer(&bufferDesc);
    bufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    bufferDesc.size = 144;
    for (auto& buffer : cameraBuffers) buffer = device.CreateBuffer(&bufferDesc);
    wgpu::BindGroupLayoutEntry entries[] {
        {.binding = 0, .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
        {.binding = 1, .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
         .buffer = {.type = wgpu::BufferBindingType::Uniform}},
    };
    wgpu::BindGroupLayoutDescriptor layoutDesc {.entryCount = 2, .entries = entries};
    auto layout = device.CreateBindGroupLayout(&layoutDesc);
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc {.bindGroupLayoutCount = 1, .bindGroupLayouts = &layout};
    auto pipelineLayout = device.CreatePipelineLayout(&pipelineLayoutDesc);
    for (int i = 0; i < 4; i++) {
        wgpu::BindGroupEntry bindings[] {
            {.binding = 0, .buffer = shapeBuffer, .size = sizeof(Gaussian)},
            {.binding = 1, .buffer = cameraBuffers[i], .size = 144},
        };
        wgpu::BindGroupDescriptor groupDesc {.layout = layout, .entryCount = 2, .entries = bindings};
        groups[i] = device.CreateBindGroup(&groupDesc);
    }
    auto shader = readWgslFile(device, SRC_DIR "/../experiments/ellipsoid.wgsl");
    wgpu::BlendState blend {
        .color = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.srcFactor = wgpu::BlendFactor::One, .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
    };
    wgpu::ColorTargetState target {.format = config.format};
    wgpu::FragmentState fragment {.module = shader, .targetCount = 1, .targets = &target};
    wgpu::DepthStencilState depth {.format = depthFormat, .depthWriteEnabled = true,
                                   .depthCompare = wgpu::CompareFunction::Less};
    wgpu::ConstantEntry falloff {.key = "gaussian_falloff", .value = 0};
    wgpu::RenderPipelineDescriptor desc {
        .layout = pipelineLayout,
        .vertex = {.module = shader},
        .depthStencil = &depth,
        .fragment = &fragment,
    };
    for (int i = 0; i < 4; i++) {
        desc.vertex.entryPoint = i == 0 ? "vs_box" : i == 1 ? "vs_quad" : "vs_splat";
        desc.primitive.topology = i == 0 ? wgpu::PrimitiveTopology::TriangleList : wgpu::PrimitiveTopology::TriangleStrip;
        fragment.entryPoint = i < 2 ? "fs_surface" : "fs_splat";
        desc.vertex.constantCount = fragment.constantCount = 1;
        desc.vertex.constants = fragment.constants = &falloff;
        falloff.value = i == 3;
        target.blend = i == 3 ? &blend : nullptr;
        depth.depthWriteEnabled = i < 2;
        pipelines[i] = device.CreateRenderPipeline(&desc);
    }
    target.blend = nullptr;
    depth.depthWriteEnabled = false;
    depth.depthCompare = wgpu::CompareFunction::Always;
    desc.primitive.topology = wgpu::PrimitiveTopology::LineList;
    fragment.entryPoint = "fs_wire";
    desc.vertex.constantCount = fragment.constantCount = 0;
    for (int i = 0; i < 2; i++) {
        desc.vertex.entryPoint = i == 0 ? "vs_box_wire" : "vs_quad_wire";
        wires[i] = device.CreateRenderPipeline(&desc);
    }

    initImgui(device, window, config.format, depthFormat);
}

void EllipsoidExperiment::draw() {
    glfwPollEvents();
    if (!handleWindowResize(surface, window, config)) return;
    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    auto& io = ImGui::GetIO();
    float uiScale = ImGui::GetStyle().FontScaleDpi;
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({io.DisplaySize.x, 112 * uiScale});
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);
    ImGui::PushItemWidth(240 * uiScale);
    ImGui::SliderFloat3("Semi-axes", scale.data(), 0.05f, 1.5f, "%.2f");
    ImGui::SameLine();
    ImGui::SliderFloat3("Rotation", rotation.data(), -180, 180, "%.0f");
    ImGui::SliderFloat("FOV", &camera.fovy, 20, 100, "%.0f");
    ImGui::SameLine();
    ImGui::SliderFloat("Opacity", &opacity, 0, 1);
    ImGui::Checkbox("Proxy outlines", &showBounds);
    ImGui::SameLine();
    if (ImGui::Button("Sphere")) scale = Eigen::Vector3f::Constant(0.7f);
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        scale = {0.9f, 0.5f, 0.3f};
        rotation = {20, 35, 15};
        camera.distance = 5;
        camera.yaw = 10;
        camera.pitch = 5;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Drag any view to orbit; scroll to zoom.");
    ImGui::PopItemWidth();
    ImGui::End();
    Eigen::Vector2d cursor;
    glfwGetCursorPos(window, &cursor.x(), &cursor.y());
    if (!io.WantCaptureMouse) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
            camera.orbit((cursor - mouse).cast<float>());
        camera.zoom(scroll);
    }
    mouse = cursor;
    scroll = 0;

    Eigen::Vector3f radians = rotation * std::numbers::pi_v<float> / 180;
    Eigen::Quaternionf q = Eigen::AngleAxisf(radians.z(), Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(radians.y(), Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(radians.x(), Eigen::Vector3f::UnitX());
    Gaussian shape {{0, 0, 0, 1}, {scale.x(), scale.y(), scale.z(), 0}, q.coeffs(), {0.25f, 0.55f, 0.9f, opacity}};
    auto queue = device.GetQueue();
    queue.WriteBuffer(shapeBuffer, 0, &shape, sizeof(shape));
    const char* titles[] {"1  Box + ellipsoid", "2  Quad + ellipsoid", "3  Projected ellipse", "4  Gaussian splat"};
    const char* subtitles[] {"36 vertices / exact ray intersection", "4 vertices / exact bounds + ray intersection",
                             "4 vertices / local perspective approximation / 1 sigma", "4 vertices / Gaussian opacity / 3 sigma cutoff"};
    const uint32_t header = std::min(uint32_t(112 * uiScale * io.DisplayFramebufferScale.y), config.height - 1);
    const uint32_t widths[] {config.width / 2, config.width - config.width / 2};
    const uint32_t heights[] {(config.height - header) / 2, config.height - header - (config.height - header) / 2};
    for (int i = 0; i < 4; i++) {
        Eigen::Matrix4f vp = camera.viewProjection(float(widths[i % 2]) / std::max(1u, heights[i / 2]));
        Eigen::Matrix4f inverse = vp.inverse();
        Eigen::Vector4f eye;
        eye << camera.position(), 1;
        queue.WriteBuffer(cameraBuffers[i], 0, vp.data(), 64);
        queue.WriteBuffer(cameraBuffers[i], 64, eye.data(), 16);
        queue.WriteBuffer(cameraBuffers[i], 80, inverse.data(), 64);
        float x = (i % 2 ? widths[0] : 0) / io.DisplayFramebufferScale.x;
        float y = (header + (i / 2 ? heights[0] : 0)) / io.DisplayFramebufferScale.y;
        auto* draw = ImGui::GetBackgroundDrawList();
        draw->AddText({x + 14 * uiScale, y + 12 * uiScale}, IM_COL32(230, 235, 245, 255), titles[i]);
        draw->AddText({x + 14 * uiScale, y + 32 * uiScale}, IM_COL32(145, 160, 180, 255), subtitles[i]);
        draw->AddRect({x, y}, {x + widths[i % 2] / io.DisplayFramebufferScale.x,
                              y + heights[i / 2] / io.DisplayFramebufferScale.y}, IM_COL32(70, 80, 95, 255));
    }
    ImGui::Render();
    if (!depthTexture || depthTexture.GetWidth() != config.width || depthTexture.GetHeight() != config.height) {
        if (depthTexture) depthTexture.Destroy();
        wgpu::TextureDescriptor desc {.usage = wgpu::TextureUsage::RenderAttachment,
                                     .size = {config.width, config.height, 1}, .format = depthFormat};
        depthTexture = device.CreateTexture(&desc);
    }
    wgpu::SurfaceTexture frame;
    surface.GetCurrentTexture(&frame);
    if (!frame.texture) return;
    wgpu::RenderPassColorAttachment color {.view = frame.texture.CreateView(), .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store, .clearValue = {0.035, 0.045, 0.065, 1}};
    wgpu::RenderPassDepthStencilAttachment depth {.view = depthTexture.CreateView(), .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Discard, .depthClearValue = 1};
    wgpu::RenderPassDescriptor desc {.colorAttachmentCount = 1, .colorAttachments = &color, .depthStencilAttachment = &depth};
    auto encoder = device.CreateCommandEncoder();
    auto pass = encoder.BeginRenderPass(&desc);
    for (int i = 0; i < 4; i++) {
        uint32_t x = i % 2 ? widths[0] : 0;
        uint32_t y = header + (i / 2 ? heights[0] : 0);
        uint32_t w = widths[i % 2], h = heights[i / 2];
        if (!w || !h) continue;
        pass.SetViewport(x, y, w, h, 0, 1);
        pass.SetScissorRect(x, y, w, h);
        pass.SetBindGroup(0, groups[i]);
        pass.SetPipeline(pipelines[i]);
        pass.Draw(i == 0 ? 36 : 4);
        if (showBounds && i < 2) {
            pass.SetPipeline(wires[i]);
            pass.Draw(i == 0 ? 24 : 8);
        }
    }
    pass.SetViewport(0, 0, config.width, config.height, 0, 1);
    pass.SetScissorRect(0, 0, config.width, config.height);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass.Get());
    pass.End();
    auto commands = encoder.Finish();
    queue.Submit(1, &commands);
#ifndef __EMSCRIPTEN__
    surface.Present();
#endif
    instance.ProcessEvents();
}

void EllipsoidExperiment::shutdown() {
    shutdownImgui();
    shutdownWgpu(surface);
    shutdownGlfw(window);
}

int main(int argc, char* argv[]) {
    static EllipsoidExperiment app {};
    app.init();
    if (argc > 1 && std::string_view(argv[1]) == "--capture") {
        app.draw();
        app.shutdown();
        return 0;
    }
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg([](void* arg) { static_cast<EllipsoidExperiment*>(arg)->draw(); }, &app, 0, false);
#else
    while (!glfwWindowShouldClose(app.window)) app.draw();
    app.shutdown();
#endif
}

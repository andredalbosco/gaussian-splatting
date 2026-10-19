#include "app.hpp"
#include "glfw_utils.hpp"
#include "imgui_utils.hpp"
#include "wgpu_utils.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

bool App::init(const char* title, int width, int height) {
    initGlfw(width, height, title, window);
    initWgpu(window, instance, device, surface, surfaceConfig);
    if (!device) return false;

    queue = device.GetQueue();
    glfwSetWindowUserPointer(window, this);
    glfwSetScrollCallback(window, [](GLFWwindow* window, double x, double y) {
        static_cast<App*>(glfwGetWindowUserPointer(window))->onScroll(x, y);
    });
    initImgui(device, window, surfaceConfig.format);
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    return true;
}

void App::run() {
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg([](void* arg) {
        static_cast<App*>(arg)->frame();
    }, this, 0, false);
#else
    while (!glfwWindowShouldClose(window)) {
        frame();
    }
    shutdown();
#endif
}

void App::shutdown() {
    shutdownImgui();
    shutdownWgpu(surface);
    shutdownGlfw(window);
}

void App::frame() {
    glfwPollEvents();
    if (!handleWindowResize(surface, window, surfaceConfig)) return;

    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    buildUi();
    update();
    ImGui::Render();

    wgpu::SurfaceTexture frame;
    surface.GetCurrentTexture(&frame);
    if (!frame.texture) return;

    wgpu::TextureView target = frame.texture.CreateView();
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    recordCommands(encoder, target);

    wgpu::RenderPassColorAttachment color {
        .view = target,
        .loadOp = wgpu::LoadOp::Load,
        .storeOp = wgpu::StoreOp::Store,
    };
    wgpu::RenderPassDescriptor passDesc {
        .colorAttachmentCount = 1,
        .colorAttachments = &color,
    };

    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass.Get());
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
#ifndef __EMSCRIPTEN__
    surface.Present();
#endif
    instance.ProcessEvents();
}

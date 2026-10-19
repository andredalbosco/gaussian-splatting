#include "imgui_utils.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>

static void scaleImgui(GLFWwindow* window) {
    float scale = ImGui_ImplGlfw_GetContentScaleForWindow(window);
    auto& style = ImGui::GetStyle();
    style = ImGuiStyle();
    ImGui::StyleColorsDark();
    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;
}

void initImgui(
    const wgpu::Device& device,
    GLFWwindow* window,
    wgpu::TextureFormat colorFormat,
    wgpu::TextureFormat depthFormat
) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui_ImplGlfw_InitForOther(window, true);
    scaleImgui(window);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#else
    glfwSetWindowContentScaleCallback(window, [](GLFWwindow* window, float, float) {
        scaleImgui(window);
    });
#endif
    ImGui_ImplWGPU_InitInfo info;
    info.Device = device.Get();
    info.RenderTargetFormat = static_cast<WGPUTextureFormat>(colorFormat);
    info.DepthStencilFormat = static_cast<WGPUTextureFormat>(depthFormat);
    ImGui_ImplWGPU_Init(&info);
}

void shutdownImgui() {
#ifndef __EMSCRIPTEN__
    auto* window = static_cast<GLFWwindow*>(ImGui::GetMainViewport()->PlatformHandle);
    glfwSetWindowContentScaleCallback(window, nullptr);
#endif
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

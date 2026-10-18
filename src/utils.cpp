#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_wgpu.h>
#include <webgpu/webgpu_glfw.h>

#ifdef __EMSCRIPTEN__
#include <GLFW/emscripten_glfw3.h>
#endif

void initGlfw(int width, int height, const char* title, GLFWwindow*& window) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#ifndef __EMSCRIPTEN__
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
#endif
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
#ifdef __EMSCRIPTEN__
    emscripten::glfw3::MakeCanvasResizable(window, "window");
#endif
}

void shutdownGlfw(GLFWwindow* window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}

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

wgpu::Instance createInstance() {
    wgpu::InstanceFeatureName feature = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor instanceDesc {
        .requiredFeatureCount = 1,
        .requiredFeatures = &feature,
    };
    return wgpu::CreateInstance(&instanceDesc);
}

wgpu::Surface createSurface(const wgpu::Instance& instance, GLFWwindow* window) {
    return wgpu::glfw::CreateSurfaceForWindow(instance, window);
}

wgpu::Adapter requestAdapter(const wgpu::Instance& instance, const wgpu::Surface& surface) {
    wgpu::RequestAdapterOptions adapterOptions {
        .powerPreference = wgpu::PowerPreference::HighPerformance,
        .compatibleSurface = surface,
    };
    wgpu::Adapter adapter;

    wgpu::Future adapterFuture = instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestAdapterStatus, wgpu::Adapter result,
           wgpu::StringView message, wgpu::Adapter* adapter) {
            *adapter = std::move(result);
            if (!*adapter) std::cerr << std::string_view(message.data, message.length) << '\n';
        },
        &adapter);

    instance.WaitAny(adapterFuture, UINT64_MAX);
    return adapter;
}

wgpu::Device requestDevice(const wgpu::Instance& instance, const wgpu::Adapter& adapter) {
    wgpu::Device device;
    wgpu::DeviceDescriptor deviceDesc {};
    deviceDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message) {
            std::cerr << std::string_view(message.data, message.length) << '\n';
        });

    wgpu::Future deviceFuture = adapter.RequestDevice(
        &deviceDesc, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::RequestDeviceStatus, wgpu::Device result,
           wgpu::StringView message, wgpu::Device* device) {
            *device = std::move(result);
            if (!*device) std::cerr << std::string_view(message.data, message.length) << '\n';
        },
        &device);

    instance.WaitAny(deviceFuture, UINT64_MAX);
    return device;
}

wgpu::SurfaceConfiguration configureSurface(
    const wgpu::Device& device,
    const wgpu::Adapter& adapter,
    const wgpu::Surface& surface,
    GLFWwindow* window
) {
    wgpu::SurfaceCapabilities capabilities;
    surface.GetCapabilities(adapter, &capabilities);

    wgpu::SurfaceConfiguration surfaceConfig {
        .device = device,
        .format = capabilities.formats[0],
        .alphaMode = capabilities.alphaModes[0],
        .presentMode = wgpu::PresentMode::Fifo,
    };
    handleWindowResize(surface, window, surfaceConfig);
    return surfaceConfig;
}

void initWgpu(
    GLFWwindow* window,
    wgpu::Instance& instance,
    wgpu::Device& device,
    wgpu::Surface& surface,
    wgpu::SurfaceConfiguration& surfaceConfig
) {
    instance = createInstance();
    surface = createSurface(instance, window);
    wgpu::Adapter adapter = requestAdapter(instance, surface);
    if (!adapter) return;
    device = requestDevice(instance, adapter);
    if (!device) return;
    surfaceConfig = configureSurface(device, adapter, surface, window);
}

void shutdownWgpu(wgpu::Surface& surface) {
    surface.Unconfigure();
    surface = nullptr;
}

Extent2D getFramebufferSize(GLFWwindow* window) {
    int width;
    int height;
    glfwGetFramebufferSize(window, &width, &height);
    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

bool checkFramebufferSize(Extent2D size) {
    if (size.width == 0 || size.height == 0) {
#ifndef __EMSCRIPTEN__
        glfwWaitEvents();
#endif
        return false;
    }

    return true;
}

bool handleWindowResize(
    const wgpu::Surface& surface,
    GLFWwindow* window,
    wgpu::SurfaceConfiguration& surfaceConfig
) {
    Extent2D size = getFramebufferSize(window);
    if (!checkFramebufferSize(size)) return false;

    if (surfaceConfig.width != size.width || surfaceConfig.height != size.height) {
        surfaceConfig.width = size.width;
        surfaceConfig.height = size.height;
        surface.Configure(&surfaceConfig);
    }

    return true;
}

std::string readFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

wgpu::ShaderModule readWgslFile(const wgpu::Device& device, const char* path) {
    std::string code = readFile(path);
    wgpu::ShaderSourceWGSL source {
        wgpu::ShaderSourceWGSL::Init {
            .code = {code.data(), code.size()},
        }
    };

    wgpu::ShaderModuleDescriptor shaderDesc {
        .nextInChain = &source,
    };

    return device.CreateShaderModule(&shaderDesc);
}

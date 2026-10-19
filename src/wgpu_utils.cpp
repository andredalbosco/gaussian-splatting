#include "wgpu_utils.hpp"
#include "utils.hpp"
#include "glfw_utils.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include <webgpu/webgpu_glfw.h>

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

#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <Eigen/Geometry>
#include <GLFW/glfw3.h>

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#endif

#ifdef __EMSCRIPTEN__
#include <GLFW/emscripten_glfw3.h>
#else
#include <webgpu/webgpu_glfw.h>
#endif

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
static void saveWindowState(GLFWwindow* window) {
    int x, y, width, height;
    glfwGetWindowPos(window, &x, &y);
    glfwGetWindowSize(window, &width, &height);
    if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) return;

    int count;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    GLFWmonitor* display = glfwGetPrimaryMonitor();
    int largestArea = 0;
    for (int i = 0; i < count; ++i) {
        int mx, my;
        glfwGetMonitorPos(monitors[i], &mx, &my);
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        int overlapWidth = std::max(0, std::min(x + width, mx + mode->width) - std::max(x, mx));
        int overlapHeight = std::max(0, std::min(y + height, my + mode->height) - std::max(y, my));
        int area = overlapWidth * overlapHeight;
        if (area > largestArea) {
            largestArea = area;
            display = monitors[i];
        }
    }

    int mx, my;
    glfwGetMonitorPos(display, &mx, &my);
    std::ofstream state(SOURCE_DIR "/.window-state");
    state << glfwGetCocoaMonitor(display) << ' ' << x - mx << ' ' << y - my
          << ' ' << width << ' ' << height << '\n';
}

static void restoreWindowState(GLFWwindow* window) {
    unsigned int displayId;
    int x, y, width, height;
    std::ifstream state(SOURCE_DIR "/.window-state");
    if (!(state >> displayId >> x >> y >> width >> height)) return;

    int count;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    GLFWmonitor* display = glfwGetPrimaryMonitor();
    for (int i = 0; i < count; ++i) {
        if (glfwGetCocoaMonitor(monitors[i]) == displayId) display = monitors[i];
    }

    int mx, my, wx, wy, ww, wh;
    glfwGetMonitorPos(display, &mx, &my);
    glfwGetMonitorWorkarea(display, &wx, &wy, &ww, &wh);
    width = std::min(width, ww);
    height = std::min(height, wh);
    int left, top, right, bottom;
    glfwGetWindowFrameSize(window, &left, &top, &right, &bottom);
    x = std::clamp(mx + x, wx, std::max(wx, wx + ww - width));
    y = std::clamp(my + y, wy + top, std::max(wy + top, wy + wh - height));
    glfwSetWindowSize(window, width, height);
    glfwSetWindowPos(window, x, y);
}
#endif

void initGlfw(int width, int height, const char* title, GLFWwindow*& window) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#endif
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    restoreWindowState(window);
    glfwShowWindow(window);
    glfwSetWindowPosCallback(window, [](GLFWwindow* window, int, int) { saveWindowState(window); });
    glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int, int) { saveWindowState(window); });
#endif
#ifdef __EMSCRIPTEN__
    emscripten::glfw3::MakeCanvasResizable(window, "window");
#endif
}

void shutdownGlfw(GLFWwindow* window) {
    glfwDestroyWindow(window);
    glfwTerminate();
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
#ifdef __EMSCRIPTEN__
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc {
        wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector::Init {
            .selector = "#canvas",
        }
    };

    wgpu::SurfaceDescriptor surfaceDesc {
        .nextInChain = &canvasDesc,
    };
    return instance.CreateSurface(&surfaceDesc);
#else
    return wgpu::glfw::CreateSurfaceForWindow(instance, window);
#endif
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

std::vector<Gaussian> readPlyFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    size_t vertexCount = 0;
    std::vector<std::string> properties;

    for (std::string line; std::getline(file, line) && line != "end_header";) {
        std::istringstream header(line);
        std::string kind;
        std::string type;
        std::string name;
        header >> kind;

        if (kind == "element") {
            header >> name;
            if (name == "vertex") header >> vertexCount;
        }

        if (kind == "property") {
            header >> type >> name;
            properties.push_back(name);
        }
    }

    auto offset = [&](const char* name) {
        return std::find(properties.begin(), properties.end(), name) - properties.begin();
    };

    size_t positionOffset = offset("x");
    size_t colorOffset = offset("f_dc_0");
    size_t scaleOffset = offset("scale_0");
    size_t rotationOffset = offset("rot_0");
    size_t opacityOffset = offset("opacity");

    std::vector<float> row(properties.size());
    std::vector<Gaussian> gaussians(vertexCount);

    for (Gaussian& gaussian : gaussians) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));

        gaussian.position.head<3>() =
            Eigen::Map<const Eigen::Vector3f>(row.data() + positionOffset);
        gaussian.position.w() = 0;

        gaussian.scale.head<3>() =
            Eigen::Map<const Eigen::Vector3f>(row.data() + scaleOffset).array().exp().matrix();
        gaussian.scale.w() = 0;

        Eigen::Quaternionf rotation {
            row[rotationOffset], row[rotationOffset + 1],
            row[rotationOffset + 2], row[rotationOffset + 3],
        };
        gaussian.rotation = rotation.normalized().coeffs();

        Eigen::Vector3f color = Eigen::Map<const Eigen::Vector3f>(row.data() + colorOffset);
        gaussian.color.head<3>() =
            (0.5f + 0.2820947918f * color.array()).max(0.0f).min(1.0f).matrix();
        gaussian.color.w() = 1.0f / (1.0f + std::exp(-row[opacityOffset]));
    }

    return gaussians;
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

#pragma once

#include <webgpu/webgpu_cpp.h>

struct GLFWwindow;

class App {
public:
    virtual ~App() = default;

    bool init(const char* title, int width = 800, int height = 600);
    void run();
    void shutdown();
    void drawFrame();

protected:
    GLFWwindow* window;
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::SurfaceConfiguration surfaceConfig;

    virtual void update() {}
    virtual void onScroll(double x, double y) {}
    virtual void drawUi() {}
    virtual void draw(const wgpu::CommandEncoder& encoder, const wgpu::TextureView& target) = 0;
};

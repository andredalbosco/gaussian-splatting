#include "glfw_utils.hpp"

#include <GLFW/glfw3.h>

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

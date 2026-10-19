#pragma once

#include <cstdint>

struct GLFWwindow;

struct Extent2D {
    uint32_t width;
    uint32_t height;
};

void initGlfw(int width, int height, const char* title, GLFWwindow*& window);
void shutdownGlfw(GLFWwindow* window);
Extent2D getFramebufferSize(GLFWwindow* window);
bool checkFramebufferSize(Extent2D size);

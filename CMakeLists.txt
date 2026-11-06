cmake_minimum_required(VERSION 3.22)
project(gaussian_splatting LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "")
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "")
set(DAWN_BUILD_TESTS OFF CACHE BOOL "")
set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "")
set(TINT_BUILD_TESTS OFF CACHE BOOL "")
set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "")

if(EMSCRIPTEN)
    add_library(glfw INTERFACE)
    target_compile_options(glfw INTERFACE --use-port=contrib.glfw3)
    target_link_options(glfw INTERFACE --use-port=contrib.glfw3)
    set(DAWN_USE_GLFW OFF CACHE BOOL "" FORCE)
else()
    set(DAWN_USE_GLFW ON CACHE BOOL "")
endif()

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "")

add_subdirectory(external/eigen EXCLUDE_FROM_ALL)
add_subdirectory(external/dawn EXCLUDE_FROM_ALL)

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_wgpu.cpp
)
target_include_directories(imgui PUBLIC external/imgui external/imgui/backends)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_DAWN GLFW_INCLUDE_NONE)
target_compile_options(imgui PRIVATE -fno-exceptions)
target_link_libraries(imgui PUBLIC glfw)
if(APPLE AND NOT EMSCRIPTEN)
    set_source_files_properties(external/imgui/backends/imgui_impl_wgpu.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()

add_executable(gaussian_splatting src/main.cpp src/utils.cpp src/gaussian.cpp src/camera.cpp)
target_compile_definitions(gaussian_splatting PRIVATE GLFW_INCLUDE_NONE)
target_compile_options(gaussian_splatting PRIVATE -fno-exceptions)
target_link_libraries(gaussian_splatting PRIVATE imgui Eigen3::Eigen)

if(EMSCRIPTEN)
    target_compile_definitions(gaussian_splatting PRIVATE SOURCE_DIR=".")
    target_link_libraries(gaussian_splatting PRIVATE emdawnwebgpu_cpp)
    target_link_libraries(imgui PUBLIC emdawnwebgpu_cpp)
    target_link_options(gaussian_splatting PRIVATE
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl@/src/shaders.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/assets/luigi.ply@/assets/luigi.ply"
    )
    set_target_properties(gaussian_splatting PROPERTIES OUTPUT_NAME "index" SUFFIX ".html")
    set_property(TARGET gaussian_splatting APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/index.html"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/luigi.ply")
else()
    target_sources(gaussian_splatting PRIVATE src/benchmark.cpp)
    target_compile_definitions(gaussian_splatting PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(gaussian_splatting PRIVATE dawn::webgpu_dawn webgpu_glfw)
    target_link_libraries(imgui PUBLIC dawn::webgpu_dawn)
endif()

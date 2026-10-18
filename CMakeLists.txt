cmake_minimum_required(VERSION 3.22)
project(gaussian_splatting LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "")
set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "")
set(DAWN_BUILD_TESTS OFF CACHE BOOL "")
set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "")
set(TINT_BUILD_TESTS OFF CACHE BOOL "")
set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "")

set(DAWN_USE_GLFW ON CACHE BOOL "" FORCE)
if(EMSCRIPTEN)
    add_library(glfw INTERFACE)
    target_compile_options(glfw INTERFACE --use-port=contrib.glfw3)
    target_link_options(glfw INTERFACE --use-port=contrib.glfw3)
endif()

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "")

add_subdirectory(external/eigen EXCLUDE_FROM_ALL)
add_subdirectory(external/dawn EXCLUDE_FROM_ALL)

if(EMSCRIPTEN)
    # The pinned Dawn GLFW helper uses DAWN_PLATFORM_IS before including its header.
    target_compile_options(dawn_glfw PRIVATE -include src/utils/platform.h)
endif()

add_compile_options(-O3)

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

add_executable(gaussian_splatting src/main.cpp src/utils.cpp src/gaussian.cpp src/camera.cpp)
target_compile_definitions(gaussian_splatting PRIVATE GLFW_INCLUDE_NONE)
target_compile_options(gaussian_splatting PRIVATE -fno-exceptions)
target_link_libraries(gaussian_splatting PRIVATE imgui Eigen3::Eigen webgpu_glfw)

if(EMSCRIPTEN)
    target_compile_definitions(gaussian_splatting PRIVATE SOURCE_DIR=".")
    target_link_libraries(gaussian_splatting PRIVATE emdawnwebgpu_cpp)
    target_link_libraries(imgui PUBLIC emdawnwebgpu_cpp)
    target_link_options(gaussian_splatting PRIVATE
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/src/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl@/src/shaders.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/assets/luigi.ply@/assets/luigi.ply"
    )
    set_target_properties(gaussian_splatting PROPERTIES OUTPUT_NAME "index" SUFFIX ".html")
    set_property(TARGET gaussian_splatting APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/index.html"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/luigi.ply")
else()
    target_compile_definitions(gaussian_splatting PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(gaussian_splatting PRIVATE dawn::webgpu_dawn)
    target_link_libraries(imgui PUBLIC dawn::webgpu_dawn)
endif()

add_executable(ellipsoid experiments/ellipsoid.cpp src/utils.cpp src/camera.cpp)
target_include_directories(ellipsoid PRIVATE src)
target_compile_definitions(ellipsoid PRIVATE GLFW_INCLUDE_NONE)
target_compile_options(ellipsoid PRIVATE -fno-exceptions)
target_link_libraries(ellipsoid PRIVATE imgui Eigen3::Eigen webgpu_glfw)
if(EMSCRIPTEN)
    target_compile_definitions(ellipsoid PRIVATE SOURCE_DIR=".")
    target_link_libraries(ellipsoid PRIVATE emdawnwebgpu_cpp)
    target_link_options(ellipsoid PRIVATE
        -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/src/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/experiments/ellipsoid.wgsl@/experiments/ellipsoid.wgsl")
    set_target_properties(ellipsoid PROPERTIES SUFFIX ".html")
else()
    target_compile_definitions(ellipsoid PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(ellipsoid PRIVATE dawn::webgpu_dawn)
endif()

if(NOT EMSCRIPTEN)
    add_executable(ellipsoid_benchmark experiments/ellipsoid_benchmark.cpp src/utils.cpp src/gaussian.cpp src/camera.cpp)
    target_include_directories(ellipsoid_benchmark PRIVATE src)
    target_compile_definitions(ellipsoid_benchmark PRIVATE GLFW_INCLUDE_NONE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(ellipsoid_benchmark PRIVATE -fno-exceptions)
    target_link_libraries(ellipsoid_benchmark PRIVATE imgui Eigen3::Eigen dawn::webgpu_dawn webgpu_glfw glfw)
endif()

if(NOT EMSCRIPTEN)
    add_executable(validate_splats experiments/validate_splats.cpp src/utils.cpp src/camera.cpp)
    target_include_directories(validate_splats PRIVATE src)
    target_compile_definitions(validate_splats PRIVATE GLFW_INCLUDE_NONE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(validate_splats PRIVATE -fno-exceptions)
    target_link_libraries(validate_splats PRIVATE imgui Eigen3::Eigen dawn::webgpu_dawn webgpu_glfw glfw)
endif()

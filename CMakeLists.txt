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

add_library(wgpu_utils STATIC src/wgpu_utils.cpp src/glfw_utils.cpp src/utils.cpp)
target_include_directories(wgpu_utils PUBLIC src)
target_compile_definitions(wgpu_utils PUBLIC GLFW_INCLUDE_NONE)
target_compile_options(wgpu_utils PRIVATE -fno-exceptions)
target_link_libraries(wgpu_utils PUBLIC webgpu_glfw glfw)
if(EMSCRIPTEN)
    target_link_libraries(wgpu_utils PUBLIC emdawnwebgpu_cpp)
else()
    target_link_libraries(wgpu_utils PUBLIC dawn::webgpu_dawn)
endif()
target_link_libraries(imgui PUBLIC wgpu_utils)

add_library(app STATIC src/app.cpp src/imgui_utils.cpp)
target_compile_options(app PRIVATE -fno-exceptions)
target_link_libraries(app PUBLIC imgui wgpu_utils)

add_library(tiled STATIC experiments/tiled.cpp)
target_include_directories(tiled PUBLIC experiments)
target_link_libraries(tiled PUBLIC wgpu_utils Eigen3::Eigen)
if(EMSCRIPTEN)
    target_compile_definitions(tiled PRIVATE SRC_DIR="./src")
else()
    target_compile_definitions(tiled PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

add_executable(gaussian_splatting src/main.cpp src/gaussian.cpp src/camera.cpp)
target_compile_definitions(gaussian_splatting PRIVATE GLFW_INCLUDE_NONE)
target_compile_options(gaussian_splatting PRIVATE -fno-exceptions)
target_link_libraries(gaussian_splatting PRIVATE app Eigen3::Eigen)

if(EMSCRIPTEN)
    target_compile_definitions(gaussian_splatting PRIVATE SRC_DIR="./src")
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
    target_compile_definitions(gaussian_splatting PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

add_executable(tiled_viewer experiments/tiled_viewer.cpp src/gaussian.cpp src/camera.cpp)
target_compile_options(tiled_viewer PRIVATE -fno-exceptions)
target_link_libraries(tiled_viewer PRIVATE app tiled Eigen3::Eigen)
if(EMSCRIPTEN)
    target_compile_definitions(tiled_viewer PRIVATE SRC_DIR="./src")
    target_link_options(tiled_viewer PRIVATE
        -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/src/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl@/src/shaders.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/experiments/tiled.wgsl@/experiments/tiled.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/experiments/tiled_present.wgsl@/experiments/tiled_present.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/assets/luigi.ply@/assets/luigi.ply")
    set_target_properties(tiled_viewer PROPERTIES SUFFIX ".html")
    set_property(TARGET tiled_viewer APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/index.html"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/experiments/tiled.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/experiments/tiled_present.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/luigi.ply")
else()
    target_compile_definitions(tiled_viewer PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

add_executable(ellipsoid experiments/ellipsoid.cpp src/camera.cpp)
target_include_directories(ellipsoid PRIVATE src)
target_compile_definitions(ellipsoid PRIVATE GLFW_INCLUDE_NONE)
target_compile_options(ellipsoid PRIVATE -fno-exceptions)
target_link_libraries(ellipsoid PRIVATE app Eigen3::Eigen)
if(EMSCRIPTEN)
    target_compile_definitions(ellipsoid PRIVATE SRC_DIR="./src")
    target_link_options(ellipsoid PRIVATE
        -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${CMAKE_CURRENT_SOURCE_DIR}/src/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/experiments/ellipsoid.wgsl@/experiments/ellipsoid.wgsl")
    set_target_properties(ellipsoid PROPERTIES SUFFIX ".html")
else()
    target_compile_definitions(ellipsoid PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

if(NOT EMSCRIPTEN)
    add_executable(ellipsoid_benchmark experiments/ellipsoid_benchmark.cpp src/gaussian.cpp src/camera.cpp)
    target_include_directories(ellipsoid_benchmark PRIVATE src)
    target_compile_definitions(ellipsoid_benchmark PRIVATE GLFW_INCLUDE_NONE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(ellipsoid_benchmark PRIVATE -fno-exceptions)
    target_link_libraries(ellipsoid_benchmark PRIVATE wgpu_utils Eigen3::Eigen)
endif()

if(NOT EMSCRIPTEN)
    add_executable(validate_splats experiments/validate_splats.cpp src/camera.cpp)
    target_include_directories(validate_splats PRIVATE src)
    target_compile_definitions(validate_splats PRIVATE GLFW_INCLUDE_NONE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(validate_splats PRIVATE -fno-exceptions)
    target_link_libraries(validate_splats PRIVATE wgpu_utils Eigen3::Eigen)
endif()

if(NOT EMSCRIPTEN)
    add_executable(validate_tiled experiments/validate_tiled.cpp src/gaussian.cpp src/camera.cpp)
    target_compile_definitions(validate_tiled PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(validate_tiled PRIVATE tiled Eigen3::Eigen)
endif()

if(NOT EMSCRIPTEN)
    add_executable(projection_benchmark experiments/projection_benchmark.cpp src/camera.cpp)
    target_compile_definitions(projection_benchmark PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(projection_benchmark PRIVATE wgpu_utils Eigen3::Eigen)
endif()

if(NOT EMSCRIPTEN)
    add_executable(space_benchmark experiments/space_benchmark.cpp src/camera.cpp)
    target_compile_definitions(space_benchmark PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(space_benchmark PRIVATE wgpu_utils Eigen3::Eigen)
endif()

if(NOT EMSCRIPTEN)
    add_executable(validate_radix_sort experiments/validate_radix_sort.cpp src/camera.cpp)
    target_compile_definitions(validate_radix_sort PRIVATE SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(validate_radix_sort PRIVATE wgpu_utils Eigen3::Eigen)
endif()

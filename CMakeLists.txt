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
    target_compile_options(dawn_glfw PRIVATE -include "${PROJECT_SOURCE_DIR}/external/dawn/src/utils/platform.h")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_compile_options(-O3 -fno-exceptions)
add_compile_definitions(GLFW_INCLUDE_NONE)
if(EMSCRIPTEN)
    add_compile_definitions(SRC_DIR="./src" ASSETS_DIR="./assets")
else()
    add_compile_definitions(SRC_DIR="${PROJECT_SOURCE_DIR}/src" ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets")
endif()

add_library(imgui STATIC
    ${PROJECT_SOURCE_DIR}/external/imgui/imgui.cpp
    ${PROJECT_SOURCE_DIR}/external/imgui/imgui_draw.cpp
    ${PROJECT_SOURCE_DIR}/external/imgui/imgui_tables.cpp
    ${PROJECT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
    ${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_glfw.cpp
    ${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_wgpu.cpp
)
target_include_directories(imgui PUBLIC ${PROJECT_SOURCE_DIR}/external/imgui ${PROJECT_SOURCE_DIR}/external/imgui/backends)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_DAWN)
target_link_libraries(imgui PUBLIC glfw)

add_library(wgpu_utils STATIC src/wgpu_utils.cpp src/glfw_utils.cpp src/utils.cpp)
target_include_directories(wgpu_utils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(wgpu_utils PUBLIC webgpu_glfw glfw)
if(EMSCRIPTEN)
    target_link_libraries(wgpu_utils PUBLIC emdawnwebgpu_cpp)
else()
    target_link_libraries(wgpu_utils PUBLIC dawn::webgpu_dawn)
endif()
target_link_libraries(imgui PUBLIC wgpu_utils)

add_library(app STATIC src/app.cpp src/imgui_utils.cpp)
target_link_libraries(app PUBLIC imgui wgpu_utils)

add_library(radix STATIC src/radix_sort.cpp)
target_link_libraries(radix PUBLIC wgpu_utils)

add_library(gaussian STATIC src/gaussian.cpp src/camera.cpp)
target_link_libraries(gaussian PUBLIC wgpu_utils Eigen3::Eigen)

add_executable(gaussian_splatting src/main.cpp)
target_link_libraries(gaussian_splatting PRIVATE app gaussian radix)

if(EMSCRIPTEN)
    target_link_options(gaussian_splatting PRIVATE
        -sASYNCIFY=1
        -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${PROJECT_SOURCE_DIR}/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl@/src/shaders.wgsl"
        "--preload-file=${PROJECT_SOURCE_DIR}/assets/black_jacket.ply@/assets/black_jacket.ply"
        "--preload-file=${PROJECT_SOURCE_DIR}/assets/luigi.ply@/assets/luigi.ply"
    )
    set_target_properties(gaussian_splatting PROPERTIES OUTPUT_NAME "index" SUFFIX ".html")
    set_property(TARGET gaussian_splatting APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.wgsl"
        "${PROJECT_SOURCE_DIR}/index.html"
        "${PROJECT_SOURCE_DIR}/assets/black_jacket.ply"
        "${PROJECT_SOURCE_DIR}/assets/luigi.ply")
endif()

add_subdirectory(experiments EXCLUDE_FROM_ALL)

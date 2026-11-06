add_library(tiled STATIC tiled.cpp)
target_include_directories(tiled PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tiled PUBLIC wgpu_utils Eigen3::Eigen)

add_executable(tiled_viewer tiled_viewer.cpp)
target_link_libraries(tiled_viewer PRIVATE app gaussian tiled)
if(EMSCRIPTEN)
    target_link_options(tiled_viewer PRIVATE
        -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${PROJECT_SOURCE_DIR}/index.html"
        "--preload-file=${PROJECT_SOURCE_DIR}/src/shaders.wgsl@/src/shaders.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/tiled.wgsl@/experiments/tiled.wgsl"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/tiled_present.wgsl@/experiments/tiled_present.wgsl"
        "--preload-file=${PROJECT_SOURCE_DIR}/assets/luigi.ply@/assets/luigi.ply")
    set_target_properties(tiled_viewer PROPERTIES SUFFIX ".html")
    set_property(TARGET tiled_viewer APPEND PROPERTY LINK_DEPENDS
        "${PROJECT_SOURCE_DIR}/index.html"
        "${PROJECT_SOURCE_DIR}/src/shaders.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/tiled.wgsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/tiled_present.wgsl"
        "${PROJECT_SOURCE_DIR}/assets/luigi.ply")
endif()

add_executable(ellipsoid ellipsoid.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp)
target_include_directories(ellipsoid PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ellipsoid PRIVATE app Eigen3::Eigen)
if(EMSCRIPTEN)
    target_link_options(ellipsoid PRIVATE
        -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1
        "--shell-file=${PROJECT_SOURCE_DIR}/index.html"
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/ellipsoid.wgsl@/experiments/ellipsoid.wgsl")
    set_target_properties(ellipsoid PROPERTIES SUFFIX ".html")
    set_property(TARGET ellipsoid APPEND PROPERTY LINK_DEPENDS
        "${PROJECT_SOURCE_DIR}/index.html"
        "${CMAKE_CURRENT_SOURCE_DIR}/ellipsoid.wgsl")
endif()

if(NOT EMSCRIPTEN)
    add_executable(validate_sh validate_sh.cpp)
    target_link_libraries(validate_sh PRIVATE gaussian radix)

    add_executable(ellipsoid_benchmark ellipsoid_benchmark.cpp)
    target_include_directories(ellipsoid_benchmark PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(ellipsoid_benchmark PRIVATE gaussian)

    add_executable(validate_splats validate_splats.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp)
    target_include_directories(validate_splats PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(validate_splats PRIVATE radix Eigen3::Eigen)

    add_executable(validate_tiled validate_tiled.cpp)
    target_link_libraries(validate_tiled PRIVATE gaussian tiled)

    add_executable(projection_benchmark projection_benchmark.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp)
    target_link_libraries(projection_benchmark PRIVATE wgpu_utils Eigen3::Eigen)

    add_executable(space_benchmark space_benchmark.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp)
    target_link_libraries(space_benchmark PRIVATE wgpu_utils Eigen3::Eigen)

    add_executable(validate_radix_sort validate_radix_sort.cpp ${PROJECT_SOURCE_DIR}/src/camera.cpp)
    target_link_libraries(validate_radix_sort PRIVATE radix Eigen3::Eigen)
endif()

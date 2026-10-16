cmake_minimum_required(VERSION 3.16)
project(gaussian_splatting LANGUAGES CXX)

add_executable(gaussian_splatting main.cpp)
target_compile_features(gaussian_splatting PRIVATE cxx_std_17)

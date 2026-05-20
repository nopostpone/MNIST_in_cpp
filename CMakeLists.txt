cmake_minimum_required(VERSION 3.16)
project(mnist_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Dependencies ──────────────────────────────────────────────────────────────

include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(Eigen)
if(NOT Eigen_POPULATED)
    FetchContent_Populate(Eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ── Sources ──────────────────────────────────────────────────────────────────

set(SOURCES
    src/main.cpp
    src/data/mnist_loader.cpp
    src/layers/conv2d.cpp
    src/layers/max_pool.cpp
    src/layers/fully_connected.cpp
    src/layers/flatten.cpp
    src/model/model.cpp
    src/model/trainer.cpp
    src/optimizer/sgd.cpp
    src/activations/relu.cpp
    src/activations/softmax.cpp
    src/loss/cross_entropy.cpp
)

# ── Executable ───────────────────────────────────────────────────────────────

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# OpenMP 加速 (可选)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MNIST_USE_OPENMP)
endif()

# ── Tests (optional) ─────────────────────────────────────────────────────────

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

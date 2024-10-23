cmake_minimum_required(VERSION 3.10)
project(GraphTsetlinMachine CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Expect NANOBIND_PATH to be set from outside (e.g., from setup.py)
if(NOT DEFINED NANOBIND_PATH)
    message(FATAL_ERROR "NANOBIND_PATH is not set. Please provide it when calling CMake.")
endif()

# Add nanobind
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_PATH}")
find_package(nanobind CONFIG REQUIRED)

# Set include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${NANOBIND_PATH}/include
)

# Add CUDA files
set(CUDA_SOURCES
    code_prepare.cu
    cuda_kernel_launcher.cu
)

# Add C++ files
set(CPP_SOURCES
    graph_tsetlin_machine_nb.cpp
)

# Detect the latest CUDA architecture
execute_process(
    COMMAND bash -c "nvcc --list-gpu-code | sort -V | tail -n1 | cut -d'_' -f2"
    OUTPUT_VARIABLE LATEST_CUDA_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Create CUDA object library
add_library(cuda_objs STATIC ${CUDA_SOURCES})
set_target_properties(cuda_objs PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(cuda_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -arch=sm_${LATEST_CUDA_ARCH}
    --compiler-options -fPIC
>)

# Create the Python module
nanobind_add_module(graphtsetlinmachine SHARED ${CPP_SOURCES})
target_link_libraries(graphtsetlinmachine PRIVATE cuda_objs ${CUDA_LIBRARIES})

# Set output name
set_target_properties(graphtsetlinmachine PROPERTIES PREFIX "")

# Installation rules
install(TARGETS graphtsetlinmachine DESTINATION .)

# Print some debug information
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "NANOBIND_PATH: ${NANOBIND_PATH}")
message(STATUS "LATEST_CUDA_ARCH: ${LATEST_CUDA_ARCH}")
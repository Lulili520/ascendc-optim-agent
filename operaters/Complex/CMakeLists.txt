cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(Complex LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# Target 1: Direct-invoke executable
add_executable(Complex op_host/Complex.asc)

target_link_libraries(Complex PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(Complex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(Complex PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

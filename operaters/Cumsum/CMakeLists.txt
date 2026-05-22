cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(Cumsum LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(Cumsum op_host/Cumsum.asc)

target_link_libraries(Cumsum PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(Cumsum PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(Cumsum PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

cmake_minimum_required(VERSION 2.8.12)

project(tc_compiled_nn)

# Directory search and filtering script
file(GLOB SUB_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_quantized)

# Automatically add _quantized directories to SUB_DIRS
list(SORT SUB_DIRS)

# Build source directory
set(BUILD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/_build_network")

foreach(dir IN LISTS SUB_DIRS)
    execute_process(COMMAND cp -f ${BUILD_SRC}/CMakeLists.txt ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
endforeach()

cmake_minimum_required(VERSION 2.8.12)

project(tc_compiled_nn)

# Directory search and filtering script
file(GLOB SUB_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_quantized)

# Automatically add _quantized directories to SUB_DIRS
list(SORT SUB_DIRS)

# Build source directory
set(BUILD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/_build_network")

foreach(dir IN LISTS SUB_DIRS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/net.so")
        # Preserve externally compiled models and their runtime ABI.
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${dir}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/net.so"
            "${CMAKE_CURRENT_BINARY_DIR}/${dir}/net.so"
            COPYONLY
        )
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${dir}/net.so"
            DESTINATION "/usr/share/${dir}")
    else()
        execute_process(COMMAND cp -f ${BUILD_SRC}/CMakeLists.txt ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
        add_subdirectory(${dir})
    endif()
endforeach()

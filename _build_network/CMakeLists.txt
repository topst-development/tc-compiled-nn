cmake_minimum_required(VERSION 2.8.12)

get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)

message(STATUS "FOLDER_NAME: ${FOLDER_NAME}")
project(${FOLDER_NAME})

set(INSTALL_PATH "/usr/share/${FOLDER_NAME}")
set(CMAKE_C_FLAGS "-Wall -D__NO_INLINE__ -std=gnu99 -lpthread -pthread -O2 -g -D__i386__ -pg -Wno-unused -Wno-attributes -fno-delete-null-pointer-checks -fPIC -no-pie")

include_directories(./)
include_directories(../_build_network)
include_directories(../_build_network/enlight)
include_directories(../_build_network/network)
include_directories(../_build_network/classifier)
include_directories(../_build_network/detector)
include_directories(../_build_network/custom_postproc)


set(SRCS
    post_process.c
    ../_build_network/classifier/classifier.c
    ../_build_network/detector/detector.c
    ../_build_network/detector/yolo_detector.c
    ../_build_network/detector/ssd_detector.c
    ../_build_network/custom_postproc/custom_postproc.c
    ../_build_network/enlight/enlight_network.c
    ../_build_network/network.c
)

add_library(${FOLDER_NAME} SHARED ${SRCS})

SET_TARGET_PROPERTIES(${FOLDER_NAME} PROPERTIES
    OUTPUT_NAME "net"
    PREFIX ""
    SUFFIX ".so"
)

install(TARGETS ${FOLDER_NAME} DESTINATION ${INSTALL_PATH})

cmake_minimum_required( VERSION 3.1 )
enable_language( C )
project( OpenCLDemo )

set( CMAKE_C_COMPILER "gcc" )
add_definitions( -std=gnu99 )

if ( CMAKE_BUILD_TYPE STREQUAL "Release" )
  add_definitions(
    -O2
    -mtune=native
    -march=native
    )
endif ( CMAKE_BUILD_TYPE STREQUAL "Release" )
if ( CMAKE_BUILD_TYPE STREQUAL "Debug" )
  add_definitions(
    -Og
    -ggdb
    -Wall
    -Wextra
    -Wpedantic
    )
endif ( CMAKE_BUILD_TYPE STREQUAL "Debug" )

file( GLOB_RECURSE HDRS
  ${PROJECT_SOURCE_DIR}/src/*.hpp
  ${PROJECT_SOURCE_DIR}/src/*.h
  )
file( GLOB_RECURSE SRCS
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/*.c
  )

configure_file( configuration/root_directory.h.in configuration/root_directory.h )
include_directories( ${CMAKE_BINARY_DIR}/configuration )

add_executable( ${CMAKE_PROJECT_NAME} ${HDRS} ${SRCS} )

find_package( OpenCL REQUIRED )
include_directories( ${OpenCL_INCLUDE_DIRS} )
link_directories( ${OpenCL_LIBRARY} )
target_include_directories( ${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

target_link_libraries( ${CMAKE_PROJECT_NAME}
  ${OpenCL_LIBRARY}
  m
)

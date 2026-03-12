cmake_minimum_required(VERSION 3.10)
project("whisper_lite" C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(ggml)

file(GLOB SOURCE_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(whisper_lite ${SOURCE_FILES})

target_include_directories(whisper_lite
    PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src   # ggml-impl.h etc.
)

target_link_libraries(whisper_lite PRIVATE ggml)

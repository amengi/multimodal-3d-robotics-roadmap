cmake_minimum_required(VERSION 3.20)

project(day019_cmake_divergence VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are disabled. Use: cmake -S . -B /tmp/day019-build")
endif()

add_library(divergence STATIC
  src/divergence.cpp
)

target_include_directories(divergence
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(divergence PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(divergence PRIVATE /W4 /WX)
else()
  target_compile_options(divergence PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(divergence_demo app/main.cpp)
target_link_libraries(divergence_demo PRIVATE divergence)

include(CTest)
if(BUILD_TESTING)
  add_executable(divergence_tests tests/test_divergence.cpp)
  target_link_libraries(divergence_tests PRIVATE divergence)
  add_test(NAME divergence_unit COMMAND divergence_tests)
endif()

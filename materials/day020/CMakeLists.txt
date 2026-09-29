cmake_minimum_required(VERSION 3.20)

project(day020_eigen_mi VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are disabled. Use: cmake -S . -B /tmp/day020-build")
endif()

set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Directory containing Eigen/Dense")
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
  find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Dense
    HINTS ${EIGEN3_INCLUDE_DIR}
    PATH_SUFFIXES eigen3
  )
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR
      "Eigen3 was not found. Set -DEIGEN3_INCLUDE_DIR=/path/containing/Eigen")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif()

add_library(day020_core STATIC src/analysis.cpp)
target_include_directories(day020_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(day020_core PUBLIC cxx_std_17)
target_link_libraries(day020_core PUBLIC Eigen3::Eigen)

if(MSVC)
  target_compile_options(day020_core PRIVATE /W4 /WX)
else()
  target_compile_options(day020_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(eigen_mi_demo app/main.cpp)
target_link_libraries(eigen_mi_demo PRIVATE day020_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(day020_tests tests/test_analysis.cpp)
  target_link_libraries(day020_tests PRIVATE day020_core)
  add_test(NAME day020_unit COMMAND day020_tests)
endif()

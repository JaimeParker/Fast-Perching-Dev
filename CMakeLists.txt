cmake_minimum_required(VERSION 3.10)
project(trajectory_optimizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

# Create library for trajectory optimization
add_library(traj_opt_lib
    src/traj_opt_perching.cc
)

# Link libraries
target_link_libraries(traj_opt_lib
    ${EIGEN3_LIBRARIES}
)

# Set target properties
target_include_directories(traj_opt_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

# Create example executables
add_executable(traj_opt_example examples/basic_example.cpp)
target_link_libraries(traj_opt_example traj_opt_lib)

add_executable(perching_comparison examples/perching_comparison.cpp)
target_link_libraries(perching_comparison traj_opt_lib)

add_executable(precision_test examples/precision_test.cpp)
target_link_libraries(precision_test traj_opt_lib)

add_executable(simple_comparison examples/simple_comparison.cpp)
target_link_libraries(simple_comparison traj_opt_lib)

# Installation rules (optional)
install(TARGETS traj_opt_lib
    EXPORT traj_opt_libConfig
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT traj_opt_libConfig
    DESTINATION lib/cmake/traj_opt_lib
)
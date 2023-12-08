cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(OSQP_DEMO)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(traj_opt)

# add the executable
add_executable(OSQP_DEMO tutorial.cpp)
target_link_libraries(OSQP_DEMO PUBLIC TrajOpt)

target_include_directories(OSQP_DEMO PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

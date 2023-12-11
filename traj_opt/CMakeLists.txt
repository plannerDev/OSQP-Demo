project(TrajOpt)
find_package(Matplot++ REQUIRED)
find_package(osqp REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE ALL_LIBRARY_HDRS "src/*.h")
file(GLOB_RECURSE ALL_LIBRARY_SRCS "src/*.cpp")
file(GLOB_RECURSE ALL_DOTFILES ".*/*")
if (ALL_DOTFILES)
  list(REMOVE_ITEM ALL_LIBRARY_HDRS ${ALL_DOTFILES})
  list(REMOVE_ITEM ALL_LIBRARY_SRCS ${ALL_DOTFILES})
endif()
message("ALL_LIBRARY_HDRS: "${ALL_LIBRARY_HDRS})
message("ALL_LIBRARY_SRCS: "${ALL_LIBRARY_SRCS})

add_library (${PROJECT_NAME} SHARED ${ALL_LIBRARY_SRCS})

target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${EIGEN3_INCLUDE_DIR}")
target_link_libraries(${PROJECT_NAME} PUBLIC ${EIGEN3_LIBRARIES} osqp::osqp)
target_link_libraries(${PROJECT_NAME} PUBLIC Matplot++::matplot)
target_link_libraries(${PROJECT_NAME} PUBLIC pthread)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

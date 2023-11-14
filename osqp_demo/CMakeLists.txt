find_package(osqp REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(DEMO osqp_demo.cpp)
target_link_libraries(DEMO PUBLIC osqp::osqp Eigen3::Eigen)

target_include_directories(DEMO
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

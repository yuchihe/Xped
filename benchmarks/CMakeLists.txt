add_executable(benchmark bench.cpp)

target_link_libraries (benchmark Xped)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9) OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Adding explicit linker flag for cpp-17 std::filesystem.")
  target_link_libraries(benchmark stdc++fs)
endif()

if(XPED_USE_AD AND XPED_MATRIX_LIB STREQUAL "Eigen" AND XPED_STORAGE STREQUAL "Contiguous")
  add_executable(stripe_u1_performance stripe_u1_performance.cpp)
  target_link_libraries(stripe_u1_performance PRIVATE Xped)
endif()
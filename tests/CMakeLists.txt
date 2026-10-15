include(doctest)

add_library(test_suite OBJECT driver.cpp)
target_link_libraries (test_suite PUBLIC
  Xped
  doctest::doctest
  )

#########################
#       basis_test      #
#########################
add_executable(basis_test basis.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (basis_test test_suite)

doctest_discover_tests(basis_test TEST_PREFIX basis__)#ADD_LABELS "1")
# if (XPED_USE_MPI)
#   add_test(NAME basis COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./basis_test ${MPIEXEC_POSTFLAGS})
# else ()
#   add_test(NAME basis COMMAND basis_test)
# endif()

#########################
#       physics_test      #
#########################
add_executable(physics_test physics.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (physics_test test_suite)

doctest_discover_tests(physics_test TEST_PREFIX physics__)#ADD_LABELS "1")
# if (XPED_USE_MPI)
#   add_test(NAME physics COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./physics_test ${MPIEXEC_POSTFLAGS})
# else ()
#   add_test(NAME physics COMMAND physics_test)
# endif()

#########################
#        tree_test      #
#########################
add_executable(tree_test tree.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (tree_test test_suite)

doctest_discover_tests(tree_test TEST_PREFIX tree__)#ADD_LABELS "1")
# if(XPED_USE_MPI)
#   add_test(NAME tree COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./tree_test ${MPIEXEC_POSTFLAGS})
# else()
#   add_test(NAME tree COMMAND tree_test)
# endif()

#########################
#       tensor_test     #
#########################
add_executable(tensor_test tensor.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (tensor_test test_suite)

doctest_discover_tests(tensor_test TEST_PREFIX tensor__)#ADD_LABELS "1")
# if(XPED_USE_MPI)
#   add_test(NAME tensor COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./tensor_test ${MPIEXEC_POSTFLAGS})
# else()
#   add_test(NAME tensor COMMAND tensor_test)
# endif()

#########################
#        mps_test       #
#########################
add_executable(mps_test mps.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (mps_test test_suite)

doctest_discover_tests(mps_test TEST_PREFIX mps__)#ADD_LABELS "1")

#########################
#        ad_test       #
#########################
add_executable(ad_test ad.cpp $<TARGET_OBJECTS:test_suite>)
target_link_libraries (ad_test test_suite)

doctest_discover_tests(ad_test TEST_PREFIX ad__)#ADD_LABELS "1")

if(XPED_USE_AD AND XPED_MATRIX_LIB STREQUAL "Eigen")
  add_executable(corner_blocking_test corner_blocking.cpp)
  target_link_libraries(corner_blocking_test PRIVATE Xped)
  add_test(NAME corner_blocking COMMAND corner_blocking_test)
  add_executable(corner_ctm_test corner_ctm.cpp)
  target_link_libraries(corner_ctm_test PRIVATE Xped)
  add_test(NAME corner_ctm COMMAND corner_ctm_test)
  add_executable(energy_slicing_test energy_slicing.cpp)
  target_link_libraries(energy_slicing_test PRIVATE Xped)
  add_test(NAME energy_slicing COMMAND energy_slicing_test)
endif()

# if(XPED_USE_MPI)
#   add_test(NAME MPS COMMAND ${MPIEXEC_EXECUTABLE} --mca opal_warn_on_missing_libcuda 0 ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./mps_test ${MPIEXEC_POSTFLAGS})
# else()
#   add_test(NAME MPS COMMAND mps_test)
# endif()

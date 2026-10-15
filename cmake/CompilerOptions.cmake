set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(set_project_options project_name)
  set(MSVC_OPTIONS_RELEASE
    /bigobj
    )
  set(MSVC_OPTIONS_DEBUG
    /bigobj
    )
  set(MSVC_OPTIONS_PROFILE
    /bigobj
    )

  set(MSVC_LOPTIONS_RELEASE
    )
  set(MSVC_LOPTIONS_DEBUG
    )
  set(MSVC_LOPTIONS_PROFILE
    )
  
  set(CLANG_OPTIONS_RELEASE
    -stdlib=${USED_LIBCXX}
    -march=native
    -ferror-limit=5
    -fcolor-diagnostics
    )
  set(CLANG_OPTIONS_DEBUG
    -stdlib=${USED_LIBCXX}
#    -march=native
    -ferror-limit=5
    -fcolor-diagnostics
    )
  set(CLANG_OPTIONS_PROFILE
    -stdlib=${USED_LIBCXX}
    -O2
    -pg
#    -march=native
    -ferror-limit=5
    -fcolor-diagnostics
    )

  set(CLANG_LOPTIONS_RELEASE
    -stdlib=${USED_LIBCXX}
    )
  set(CLANG_LOPTIONS_DEBUG
    -stdlib=${USED_LIBCXX}
    )
  set(CLANG_LOPTIONS_PROFILE
    -stdlib=${USED_LIBCXX}
    -O2
    -pg
    )
  
  set(GCC_OPTIONS_RELEASE
    -march=native
    -fmax-errors=5
    -fdiagnostics-color=always
    )
  set(GCC_OPTIONS_DEBUG
    -march=native
    -fmax-errors=5
    -fdiagnostics-color=always
    )
  set(GCC_OPTIONS_PROFILE
    -O2
    -pg
    -march=native
    -fmax-errors=5
    -fdiagnostics-color=always
    )

  set(GCC_LOPTIONS_RELEASE
    )
  set(GCC_LOPTIONS_DEBUG
    )
  set(GCC_LOPTIONS_PROFILE
    -O2
    -pg
    )

  set(INTEL_OPTIONS_RELEASE
    -xHost
    )
  set(INTEL_OPTIONS_DEBUG
    )
  set(INTEL_OPTIONS_PROFILE
    -O2
    -g
    -prof-gen
    )

  set(INTEL_LOPTIONS_RELEASE
    -xHost
    )
  set(INTEL_LOPTIONS_DEBUG
    )
  set(INTEL_LOPTIONS_PROFILE
    -O2
    )

  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    if(MSVC)
      set(PROJECT_OPTIONS ${MSVC_OPTIONS_RELEASE})
      set(PROJECT_LOPTIONS ${MSVC_LOPTIONS_RELEASE})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      set(PROJECT_OPTIONS ${CLANG_OPTIONS_RELEASE})
      set(PROJECT_LOPTIONS ${CLANG_LOPTIONS_RELEASE})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(PROJECT_OPTIONS ${GCC_OPTIONS_RELEASE})
      set(PROJECT_LOPTIONS ${GCC_LOPTIONS_RELEASE})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      set(PROJECT_OPTIONS ${INTEL_OPTIONS_RELEASE})
      set(PROJECT_LOPTIONS ${INTEL_LOPTIONS_RELEASE})
    else()
      message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    if(MSVC)
      set(PROJECT_OPTIONS ${MSVC_OPTIONS_DEBUG})
      set(PROJECT_LOPTIONS ${MSVC_LOPTIONS_DEBUG})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      set(PROJECT_OPTIONS ${CLANG_OPTIONS_DEBUG})
      set(PROJECT_LOPTIONS ${CLANG_LOPTIONS_DEBUG})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(PROJECT_OPTIONS ${GCC_OPTIONS_DEBUG})
      set(PROJECT_LOPTIONS ${GCC_LOPTIONS_DEBUG})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      set(PROJECT_OPTIONS ${INTEL_OPTIONS_RELEASE})
      set(PROJECT_LOPTIONS ${INTEL_LOPTIONS_RELEASE})
    else()
      message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Profile")
    if(MSVC)
      set(PROJECT_OPTIONS ${MSVC_OPTIONS_PROFILE})
      set(PROJECT_LOPTIONS ${MSVC_LOPTIONS_PROFILE})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      set(PROJECT_OPTIONS ${CLANG_OPTIONS_PROFILE})
      set(PROJECT_LOPTIONS ${CLANG_LOPTIONS_PROFILE})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(PROJECT_OPTIONS ${GCC_OPTIONS_PROFILE})
      set(PROJECT_LOPTIONS ${GCC_LOPTIONS_PROFILE})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      set(PROJECT_OPTIONS ${INTEL_OPTIONS_RELEASE})
      set(PROJECT_LOPTIONS ${INTEL_LOPTIONS_RELEASE})
    else()
      message(AUTHOR_WARNING "No compiler options set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()
  endif()

  target_compile_options(${project_name} INTERFACE ${PROJECT_OPTIONS})

  target_link_options(${project_name} INTERFACE ${PROJECT_LOPTIONS})

  target_compile_definitions(${project_name} INTERFACE SPDLOG_ACTIVE_LEVEL=${XPED_LOG_LEVEL})

  if(XPED_COMPILED_LIB)
    target_compile_definitions(${project_name} INTERFACE XPED_COMPILED_LIB=1)
  endif()

  if(XPED_ENABLE_LRU_CACHE)
    target_compile_definitions(${project_name} INTERFACE XPED_CACHE_PERMUTE_OUTPUT=1)
  endif()

  if(XPED_USE_AD)
    target_compile_definitions(${project_name} INTERFACE XPED_USE_AD=1)
    if(XPED_USE_EXTERNAL_STAN)
      # only the external stan-math backend needs the TBB-related definitions
      target_compile_definitions(${project_name} INTERFACE _REENTRANT=1)
      target_compile_definitions(${project_name} INTERFACE TBB_INTERFACE_NEW)
    endif()
  endif()

  if(XPED_USE_GSL_COEFFS)
    target_compile_definitions(${project_name} INTERFACE XPED_USE_GSL_SU2_COEFFS=1)
  endif()

  if(${XPED_STORAGE} STREQUAL "Contiguous")
    target_compile_definitions(${project_name} INTERFACE XPED_USE_CONTIGUOUS_STORAGE=1)
  elseif(${XPED_STORAGE} STREQUAL "VecOfMat")
    target_compile_definitions(${project_name} INTERFACE XPED_USE_VECOFMAT_STORAGE=1)
  endif()
  
  if(${XPED_TENSOR_LIB} STREQUAL "Eigen")
    target_compile_definitions(${project_name} INTERFACE XPED_USE_EIGEN_TENSOR_LIB=1)
  elseif (${XPED_TENSOR_LIB} STREQUAL "Array")
    target_compile_definitions(${project_name} INTERFACE XPED_USE_ARRAY_TENSOR_LIB=1)
  elseif (${XPED_TENSOR_LIB} STREQUAL "Cyclops")  
    target_compile_definitions(${project_name} INTERFACE XPED_USE_CYCLOPS_TENSOR_LIB=1)
  endif()

  if(${XPED_MATRIX_LIB} STREQUAL "Eigen")
    target_compile_definitions(${project_name} INTERFACE XPED_USE_EIGEN_MATRIX_LIB=1)
  elseif (${XPED_MATRIX_LIB} STREQUAL "Cyclops")  
    target_compile_definitions(${project_name} INTERFACE XPED_USE_CYCLOPS_MATRIX_LIB=1)
  endif()

  if(${XPED_VECTOR_LIB} STREQUAL "Eigen")
    target_compile_definitions(${project_name} INTERFACE XPED_USE_EIGEN_VECTOR_LIB=1)
  elseif (${XPED_VECTOR_LIB} STREQUAL "Cyclops")  
    target_compile_definitions(${project_name} INTERFACE XPED_USE_CYCLOPS_VECTOR_LIB=1)
  endif()

  if(XPED_USE_NLO)
    if(${XPED_OPTIM_LIB} STREQUAL "ceres")
      target_compile_definitions(${project_name} INTERFACE XPED_USE_CERES_OPTIMIZATION=1)
    endif()
  endif()

  if(${XPED_EFFICIENCY_MODEL} STREQUAL "XPED_TIME_EFFICIENT")
    target_compile_definitions(${project_name} INTERFACE XPED_EFFICIENCY_MODEL=1)
    target_compile_definitions(${project_name} INTERFACE XPED_TIME_EFFICIENT=1)
  elseif(${XPED_EFFICIENCY_MODEL} STREQUAL "XPED_MEMORY_EFFICIENT")
    target_compile_definitions(${project_name} INTERFACE XPED_EFFICIENCY_MODEL=1)
    target_compile_definitions(${project_name} INTERFACE XPED_MEMORY_EFFICIENT=1)
  endif()

  if(XPED_USE_MPI OR XPED_MKL_USE_MPI)
    target_compile_definitions(${project_name} INTERFACE XPED_USE_MPI=1)
  endif()

  if(XPED_USE_BLAS)
    target_compile_definitions(${project_name} INTERFACE XPED_USE_BLAS=1)
    target_compile_definitions(${project_name} INTERFACE EIGEN_USE_BLAS=1)
    if(XPED_USE_LAPACK)
      target_compile_definitions(${project_name} INTERFACE EIGEN_USE_LAPACKE=1)
      # Keep BDCSVD as the default: Eigen dispatches it to LAPACK gesdd.
    endif()
  endif()

  if(XPED_USE_MKL)
    target_compile_definitions(${project_name} INTERFACE XPED_USE_MKL=1)
    if(${XPED_MATRIX_LIB} STREQUAL "Eigen")
      target_compile_definitions(${project_name} INTERFACE EIGEN_USE_BLAS=1)
      target_compile_definitions(${project_name} INTERFACE EIGEN_USE_LAPACKE=1)
      # Keep BDCSVD as the default: Eigen dispatches it to MKL gesdd.
      target_compile_definitions(${project_name} INTERFACE EIGEN_USE_MKL_VML=1)
    endif()
  endif()

  if(XPED_PEDANTIC_ASSERTS)
    target_compile_definitions(${project_name} INTERFACE XPED_PEDANTIC_ASSERTS=1)
  endif()

endfunction()

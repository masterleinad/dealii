## ---------------------------------------------------------------------
##
## Copyright (C) 2016 - 2018 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

#
# Configuration for cuda support:
#

#
# cuda support is experimental. Therefore, disable the feature per default:
#
SET(DEAL_II_WITH_CUDA FALSE CACHE BOOL "")

MACRO(FEATURE_CUDA_FIND_EXTERNAL var)
  INCLUDE(CheckLanguage)
  CHECK_LANGUAGE(CUDA)
  IF(CMAKE_CUDA_COMPILER)
    SET(${var} TRUE)
    ENABLE_LANGUAGE(CUDA)
  ELSE()
    MESSAGE(STATUS "Could not find a suitable CUDA compiler.")
    SET(CUDA_ADDITIONAL_ERROR_STRING
       SET(${var} FALSE)
       ${CUDA_ADDITIONAL_ERROR_STRING}
       "Could not find a suitable CUDA compiler.\n"
       "Please set CMAKE_CUDA_COMPILER to a working CUDA compiler." 
       )
  ENDIF()
  SET(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)

MESSAGE(STATUS "cudatoolkitrootdir: ${CUDA_TOOLKIT_ROOT_DIR}")
MESSAGE(STATUS "cudadi: ${CUDA_DIR}")

DEAL_II_FIND_LIBRARY(CUDA_cusolver_LIBRARY
NAMES cusolver libcusolver
             HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_DIR}
PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
)

#DEAL_II_FIND_LIBRARY(CUDA_cusparse_LIBRARY
#  NAMES cusparse libcusparse
#  HINTS ${CUDA_DIR}
#  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
#  )
#
#DEAL_II_FIND_LIBRARY(CUDA_cusolver_LIBRARY
#  NAMES cusolver libcusolver
#  HINTS ${CUDA_DIR}
#  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
#  )
#
#DEAL_II_FIND_PATH(CUDA_INCLUDE_DIRS cuda.h
#  HINTS ${CUDA_DIR}
#  PATH_SUFFIXES cuda include include/cuda
#)

  SET(_cuda_libraries ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})
  MESSAGE(STATUS "_cuda_libraries: ${_cuda_libraries}")
  #SET(_cuda_include_dirs ${CUDA_INCLUDE_DIRS})
  #MESSAGE(STATUS "_cuda_include_dirs: ${_cuda_include_dirs}")
  DEAL_II_PACKAGE_HANDLE(CUDA
    LIBRARIES REQUIRED _cuda_libraries
    INCLUDE_DIRS _cuda_include_dirs
    USER_INCLUDE_DIRS _cuda_include_dirs
    CLEAR
      CUDA_cublas_device_LIBRARY
      CUDA_cublas_LIBRARY
      CUDA_cudadevrt_LIBRARY
      CUDA_cudart_static_LIBRARY
      CUDA_cufft_LIBRARY
      CUDA_cupti_LIBRARY
      CUDA_curand_LIBRARY
      CUDA_HOST_COMPILER
      CUDA_nppc_LIBRARY
      CUDA_nppi_LIBRARY
      CUDA_npps_LIBRARY
      CUDA_rt_LIBRARY
      CUDA_SDK_ROOT_DIR
      CUDA_TOOLKIT_ROOT_DIR
      CUDA_USE_STATIC_CUDA_RUNTIME
  )

  IF(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    SET(${var} FALSE)
    MESSAGE(STATUS "deal.II only supports CUDA via Nvidia's nvcc wrapper")
    SET(CUDA_ADDITIONAL_ERROR_STRING
        ${CUDA_ADDITIONAL_ERROR_STRING}
        "deal.II only support CUDA via Nvidia's nvcc wrapper.\n"
        "Reconfigure with a different CUDA compiler."
        )
  ENDIF()

    #
    # CUDA support requires CMake version 3.9 or newer
    #
    IF(CMAKE_VERSION VERSION_LESS 3.9)
      SET(${var} FALSE)
      MESSAGE(STATUS "deal.II requires CMake version 3.9, or newer for CUDA support")
      SET(CUDA_ADDITIONAL_ERROR_STRING
        ${CUDA_ADDITIONAL_ERROR_STRING}
        "deal.II requires CMake version 3.9, or newer for CUDA support.\n"
        "Reconfigure with a sufficient cmake version."
        )
    ENDIF()

    message(status "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

    #
    # CUDA 8.0 does not support C++14, 
    # CUDA 9.0 and CUDA 10.0 support C++14, but not C++17.
    # Make sure that deal.II is configured appropriately
    #
    MACRO(_cuda_ensure_feature_off _version _feature)
      STRING(REGEX MATCH "^[0-9]+" CMAKE_CUDA_COMPILER_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
      IF(${CMAKE_CUDA_COMPILER_VERSION_MAJOR} EQUAL ${_version})
        IF(${_feature})
          SET(${var} FALSE)
          MESSAGE(STATUS "CUDA ${_version} requires ${_feature} to be set to off.")
          SET(CUDA_ADDITIONAL_ERROR_STRING
            ${CUDA_ADDITIONAL_ERROR_STRING}
            "CUDA ${_version} is not compatible with the C++ standard\n"
            "enabled by ${_feature}.\n"
            "Please disable ${_feature}, e.g. by reconfiguring with\n"
            "  cmake -D${_feature}=OFF ."
            )
        ENDIF()
      ENDIF()
    ENDMACRO()
    _cuda_ensure_feature_off(8 DEAL_II_WITH_CXX14)
    _cuda_ensure_feature_off(9 DEAL_II_WITH_CXX17)
    _cuda_ensure_feature_off(10 DEAL_II_WITH_CXX17)


    IF("${DEAL_II_CUDA_FLAGS_SAVED}" MATCHES "-arch[ ]*sm_([0-9]*)")
      SET(CUDA_COMPUTE_CAPABILITY "${CMAKE_MATCH_1}")
    ELSEIF("${DEAL_II_CUDA_FLAGS_SAVED}" MATCHES "-arch=sm_([0-9]*)")
      SET(CUDA_COMPUTE_CAPABILITY "${CMAKE_MATCH_1}")
    ELSE()
      #
      # Assume a cuda compute capability of 35
      #
      SET(CUDA_COMPUTE_CAPABILITY "35")
      ADD_FLAGS(DEAL_II_CUDA_FLAGS "-arch=sm_35")
    ENDIF()

    IF("${CUDA_COMPUTE_CAPABILITY}" LESS "35")
      MESSAGE(STATUS "Too low CUDA Compute Capability specified -- deal.II requires at least 3.5 ")
      SET(CUDA_ADDITIONAL_ERROR_STRING
        ${CUDA_ADDITIONAL_ERROR_STRING}
        "Too low CUDA Compute Capability specified: ${CUDA_COMPUTE_CAPABILITY}\n"
        "deal.II requires at least Compute Capability 3.5\n"
        "which is used as default if nothing is specified."
        )
      SET(${var} FALSE)
    ENDIF()

    # cuSOLVER requires OpenMP
    FIND_PACKAGE(OpenMP)
    SET(DEAL_II_LINKER_FLAGS "${DEAL_II_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")

    ADD_FLAGS(DEAL_II_CUDA_FLAGS_DEBUG "-G")
  #ENDIF()
ENDMACRO()


MACRO(FEATURE_CUDA_CONFIGURE_EXTERNAL)

  #
  # Ensure that we enable CMake-internal CUDA support with the right
  # compiler:
  #
  #SET(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
  SET(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  ENABLE_LANGUAGE(CUDA)

  #
  # Work around a cmake 3.10 bug, see https://gitlab.kitware.com/cmake/cmake/issues/17797
  # because make does not support rsp link commands
  #
  SET(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  SET(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  SET(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

  #
  # Set up cuda flags:
  #
  ADD_FLAGS(DEAL_II_CUDA_FLAGS "${DEAL_II_CXX_VERSION_FLAG}")

  # We cannot use -pedantic as compiler flags. nvcc generates code that
  # produces a lot of warnings when pedantic is enabled. So filter out the
  # flag:
  #
  STRING(REPLACE "-pedantic" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")

  #
  # Export definitions:
  #
  STRING(SUBSTRING "${CUDA_COMPUTE_CAPABILITY}" 0 1 CUDA_COMPUTE_CAPABILITY_MAJOR)
  STRING(SUBSTRING "${CUDA_COMPUTE_CAPABILITY}" 1 1 CUDA_COMPUTE_CAPABILITY_MINOR)
ENDMACRO()


MACRO(FEATURE_CUDA_ERROR_MESSAGE)
  MESSAGE(FATAL_ERROR "\n"
    "Could not find any suitable cuda library!\n"
    ${CUDA_ADDITIONAL_ERROR_STRING}
    "\nPlease ensure that a cuda library is installed on your computer\n"
    )
ENDMACRO()


CONFIGURE_FEATURE(CUDA)

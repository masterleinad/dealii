## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2018 by the deal.II authors
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

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

#
# All source/base files are sufficiently small that they may be added to the
# unity build file (see the developer documentation on DEAL_II_UNITY_BUILD
# for more information).
#

SET(_separate_src
  )

# Add CUDA wrapper files
IF(DEAL_II_WITH_CUDA)
  SET(_separate_src
    ${_separate_src}
    #logstream.cc
    #exceptions.cc
    #subscriptor.cc
    #thread_management.cc
    #job_identifier.cc
    cuda.cu
  )
ENDIF()

# determined by profiling
SET(_n_includes_per_unity_file 29)

SETUP_SOURCE_LIST("${_unity_include_src}"
  "${_separate_src}"
  ${_n_includes_per_unity_file}
  _src
  )

SET(_inst
  )

#FILE(GLOB _header
#  ${CMAKE_SOURCE_DIR}/include/deal.II/base/*.h
#  ${CMAKE_SOURCE_DIR}/include/deal.II/base/*.cuh
#  )

DEAL_II_ADD_LIBRARY(obj_base OBJECT cuda.cu
    #${_src} ${_header} ${_inst}
    #  ${CMAKE_BINARY_DIR}/include/deal.II/base/config.h
  )
#EXPAND_INSTANTIATIONS(obj_base "${_inst}")

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

#
# List the directories where we have source files. the ones with the longest
# compile jobs come first so that 'make -j N' saturates many processors also
# towards the end of compiling rather than having to wait for one long
# compilation that, because it has been listed last, is started towards the
# end of everything (e.g. numerics/vectors.cc takes several minutes to
# compile...)
#
ADD_SUBDIRECTORY(base)

  GET_PROPERTY(_objects GLOBAL PROPERTY DEAL_II_OBJECTS_DEBUG)

  ADD_LIBRARY(${DEAL_II_BASE_NAME}${DEAL_II_DEBUG_SUFFIX}
    ${_objects}
    )
  ADD_DEPENDENCIES(library ${DEAL_II_BASE_NAME}${DEAL_II_DEBUG_SUFFIX})

  SET_TARGET_PROPERTIES(${DEAL_II_BASE_NAME}${DEAL_II_DEBUG_SUFFIX}
    PROPERTIES
    VERSION "${DEAL_II_PACKAGE_VERSION}"
    #
    # Sonaming: Well... we just use the version number.
    # No point to wrack one's brain over the question whether a new version of
    # a C++ library is still ABI backwards compatible :-]
    #
    SOVERSION "${DEAL_II_PACKAGE_VERSION}"
    LINK_FLAGS "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_DEBUG}"
    LINKER_LANGUAGE "CXX"
    COMPILE_DEFINITIONS "${DEAL_II_DEFINITIONS};${DEAL_II_DEFINITIONS_DEBUG}"
    COMPILE_FLAGS "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_DEBUG}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEAL_II_EXECUTABLE_RELDIR}"
    )

  TARGET_LINK_LIBRARIES(${DEAL_II_BASE_NAME}${DEAL_II_DEBUG_SUFFIX}
    ${DEAL_II_LIBRARIES_DEBUG}
    ${DEAL_II_LIBRARIES}
    )

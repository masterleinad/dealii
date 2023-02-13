## ---------------------------------------------------------------------
##
## Copyright (C) 2016 - 2022 by the deal.II authors
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
# Usage:
#   check_compiler_setup("compiler flag string" "linker flag string" _var
#     [libraries]
#     )
#
# This macro tries to compile and link a simple "int main(){ return 0; }
# with the given set of provided compiler and linker flags  and an optional
# list of libraries to link against. If the test is successful the variable
# ${_var} is set to true, otherwise it is set to false.
#

macro(check_compiler_setup _compiler_flags_unstr _linker_flags_unstr _var)
  #
  # Strip leading and trailing whitespace to make CMake 2.8.8 happy
  #
  string(STRIP "${_compiler_flags_unstr}" _compiler_flags)
  string(STRIP "${_linker_flags_unstr}" _linker_flags)

  separate_arguments(_compiler_flags)

  #
  # Rerun this test if flags have changed:
  #
  if(NOT "${_compiler_flags}" STREQUAL "${CACHED_${_var}_compiler_flags}"
     OR NOT "${_linker_flags}" STREQUAL "${CACHED_${_var}_linker_flags}"
     OR NOT "${ARGN}" STREQUAL "${CACHED_${_var}_ARGN}")
    unset(${_var} CACHE)
  endif()

  set(CACHED_${_var}_compiler_flags "${_compiler_flags}"
    CACHE INTERNAL "" FORCE
    )
  set(CACHED_${_var}_linker_flags "${_linker_flags}"
    CACHE INTERNAL "" FORCE
    )
  set(CACHED_${_var}_ARGN "${ARGN}" CACHE INTERNAL "" FORCE)

  message(STATUS "Testing ${_var}") 

  try_compile(
    ${_var}
    ${CMAKE_CURRENT_BINARY_DIR}/check_compiler_setup/CheckCompilerSetup${_var}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros/check_compiler_setup
    CheckCompilerSetup${_var}
    CMAKE_FLAGS "-DTEST_COMPILE_FLAGS=${_compiler_flags}" 
                "-DTEST_LINK_OPTIONS=${_linker_flags}"
                "-DTEST_LINK_LIBRARIES=${ARGN}"
                "-DCMAKE_VERBOSE_MAKEFILE=ON"
    OUTPUT_VARIABLE _output)

  message(STATUS "Compiler setup check for ${_var} failed with:\n${_output}")

  if(${_var})
    set(${_var} TRUE CACHE INTERNAL "")
    file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/check_compiler_setup/CheckCompilerSetup${var})
  else()
    message(STATUS "Compiler setup check for ${_var} failed with:\n${_output}")
  endif()
endmacro()

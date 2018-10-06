CMAKE_MINIMUM_REQUIRED(VERSION 3.12.1)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules/)

#
# Load all macros:
#
FILE(GLOB _macro_files "cmake/macros/*.cmake")
MESSAGE(STATUS "Include ${CMAKE_SOURCE_DIR}/cmake/setup_external_macros.cmake")
INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_external_macros.cmake)
FOREACH(_file ${_macro_files})
  MESSAGE(STATUS "Include ${_file}")
  INCLUDE(${_file})
ENDFOREACH()

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

#
# We have to initialize some cached variables before PROJECT is called, so
# do it at this point:
#
VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_cached_variables.cmake)

#
# Now, set the project and set up the rest:
#
PROJECT(deal.II CXX C)

VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_deal_ii.cmake)

VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_compiler_flags.cmake)

#
# Run all system checks:
#
FILE(GLOB _check_files "cmake/checks/*.cmake")
LIST(SORT _check_files)
FOREACH(_file ${_check_files})
  VERBOSE_INCLUDE(${_file})
ENDFOREACH()

#
# Feature configuration:
#
FILE(GLOB _configure_files "cmake/configure/configure_*.cmake")
LIST(SORT _configure_files) # make sure to include in alphabetical order
FOREACH(_file ${_configure_files})
  VERBOSE_INCLUDE(${_file})
ENDFOREACH()

#
# Finalize the configuration:
#
VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_cpack.cmake)
VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_custom_targets.cmake)
VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_finalize.cmake)
VERBOSE_INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_write_config.cmake)

ADD_SUBDIRECTORY(source) # has to be included after bundled

ADD_SUBDIRECTORY(tests)


if("${TRACK}" STREQUAL "")
  set(TRACK "Build Tests")
endif()
set(DEAL_II_COMPILE_EXAMPLES TRUE)
set(TEST_PICKUP_REGEX "quick_tests/")
include(${CMAKE_CURRENT_LIST_DIR}/run_testsuite.cmake)

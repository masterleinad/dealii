project(CheckCompilerSetup)
cmake_minimum_required(VERSION 3.13.4)
add_executable(CheckCompilerSetupExec dummy.cpp)

message(STATUS "TEST_COMPILE_FLAGS: ${TEST_COMPILE_FLAGS}")
message(STATUS "TEST_LINK_OPTIONS: ${TEST_LINK_OPTIONS}")
message(STATUS "TEST_LINK_LIBRARIES: ${TEST_LINK_LIBRARIES}")

target_compile_options(CheckCompilerSetupExec PRIVATE ${TEST_COMPILE_FLAGS})
# Should we rather use
#   target_link_options(CheckCompilerSetupExec PRIVATE ${TEST_LINK_OPTIONS})
# and convert whitespace-separated list to comma-separated list?
set_property(TARGET CheckCompilerSetupExec PROPERTY LINK_FLAGS "${TEST_LINK_OPTIONS}")

target_link_libraries(CheckCompilerSetupExec ${TEST_LINK_LIBRARIES})


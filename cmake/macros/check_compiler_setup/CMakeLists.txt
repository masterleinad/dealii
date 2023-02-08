project(CheckCompilerSetup)
cmake_minimum_required(VERSION 3.13.4)
add_executable(CheckCompilerSetupExec dummy.cpp)

MESSAGE(STATUS "1: ${TEST_COMPILE_FLAGS}\n 2:${TEST_LINK_OPTIONS}\n${TEST_LINK_LIBRARIES}")

target_compile_options(CheckCompilerSetupExec PRIVATE "${TEST_COMPILE_FLAGS}")
target_link_options(CheckCompilerSetupExec PRIVATE "${TEST_LINK_OPTIONS}")
target_link_libraries(CheckCompilerSetupExec "${TEST_LINK_LIBRARIES}")


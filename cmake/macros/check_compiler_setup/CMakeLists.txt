project(CheckCompilerSetup)
cmake_minimum_required(VERSION 3.13.4)
add_executable(CheckCompilerSetupExec dummy.cpp)

target_compile_options(CheckCompilerSetupExec PRIVATE ${TEST_COMPILE_FLAGS})
target_link_options(CheckCompilerSetupExec PRIVATE ${TEST_LINK_OPTIONS})
target_link_libraries(CheckCompilerSetupExec ${TEST_LINK_LIBRARIES})


cmake_minimum_required(VERSION 3.14)
include(CMakePrintHelpers) # for debugging

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CastCompiler VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_INCLUDE_DIRS}")
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${CIRCT_INCLUDE_DIRS})

add_executable(castc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Frontend/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Frontend/castLower.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Frontend/CastParser.cpp
)

target_include_directories(castc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(castc
    CIRCTHW
    CIRCTSV
    CIRCTSeq
    CIRCTComb
    CIRCTFSM
    CIRCTESI
    CIRCTExportVerilog
    CIRCTSeqToSV
    CIRCTSeqTransforms
    CIRCTVerifToSV
    CIRCTSupport
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)

if(APPLE)
    target_link_options(castc PRIVATE 
        "-Wl,-no_warn_duplicate_libraries"
        "-L/Users/armv9/llvm-project/build/lib"        
        "-Wl,-rpath,/Users/armv9/llvm-project/build/lib"
    )
    # Force linking the custom libc++
    target_link_libraries(castc c++)
endif()

target_compile_options(castc PRIVATE
    -Wall -Wextra -Wpedantic
    -fexceptions
    -frtti
    -fdiagnostics-color=always
)

# ── Testing ────────────────────────────────────────────────────────────────────
# Run with:  ninja check
#   Compiles every listed example through castc (--tb) then elaborates the
#   resulting Verilog with iverilog to catch both cast-compiler errors and
#   malformed Verilog output.
# ─────────────────────────────────────────────────────────────────────────────
enable_testing()

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j1
    DEPENDS castc
    COMMENT "Running cast example tests"
)

set(CAST_TEST_RUNNER ${CMAKE_SOURCE_DIR}/tests/run_cast_test.sh)
set(CASTC_BIN        ${CMAKE_BINARY_DIR}/castc)
set(EXAMPLES_DIR     ${CMAKE_SOURCE_DIR}/examples)
set(EXPECTED_DIR     ${CMAKE_SOURCE_DIR}/tests/expected)

# add_cast_test(<name>)
#   Stage 1: castc compiles the example (--duration=100 when expected file present)
#   Stage 2: iverilog elaborates the output Verilog
#   Stage 3: vvp simulates and diffs against tests/expected/<name>.expected
#            (only runs when the expected file exists)
function(add_cast_test name)
    add_test(
        NAME    example.${name}
        COMMAND bash ${CAST_TEST_RUNNER}
                     ${CASTC_BIN}
                     ${EXAMPLES_DIR}/${name}.cast
                     ${EXPECTED_DIR}/${name}.expected
    )
    set_tests_properties(example.${name} PROPERTIES
        TIMEOUT 30
        LABELS  "cast"
    )
endfunction()

# Examples expected to compile cleanly.
# Add new examples here as features are implemented.
add_cast_test(simple)
add_cast_test(counter)
add_cast_test(fibonacci)
add_cast_test(gcd)
add_cast_test(alarm)
add_cast_test(running_average)
add_cast_test(traffic_light)
add_cast_test(checksum)
add_cast_test(debounce)
add_cast_test(shift_register)
add_cast_test(popcount)
add_cast_test(binary_search)
add_cast_test(pwm)
add_cast_test(uart_tx)
add_cast_test(test_loop)
add_cast_test(matrix_multiply)
add_cast_test(matrix_multiply_loops)
add_cast_test(array_sum)
add_cast_test(systolic_matmul)
add_cast_test(systolic_matmul_5x5_scalar)
add_cast_test(pipeline)

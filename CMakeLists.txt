# To compile this project you'll need to have CUDA Toolkit >= 5.0 installed.
# To run the compiled executables, you'll need to have an nVidia GPU with Compute Capability >= 3.5
#
# Every CUDA Toolkit release has a list of supported GCC/MSVC compilers.
# Make sure you have a compatible CUDA Toolkit and C++ compiler pair installed.
#
# In case of problems with building this project on Windows using older CUDA Toolkits and/or MSVC compilers
# try turning on the CUSTOM_NVCC_BUILD variable.
# You'll also need to adjust the variables in Compile.bat because most probably you'll have different paths than mine.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda-exercises LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CUSTOM_NVCC_BUILD false)

file(GLOB files "${CMAKE_SOURCE_DIR}/src/*.cu")
if (WIN32 AND CUSTOM_NVCC_BUILD)
    foreach (file ${files})
        # Get only the filename without the absolute path
        get_filename_component(file ${file} NAME_WE)
        # This command is responsible for recompiling the .exe if a corresponding .cu file was modified
        add_custom_command(
                OUTPUT "${CMAKE_SOURCE_DIR}/build/${file}.exe"
                COMMAND cmd /C "${CMAKE_SOURCE_DIR}/Compile.bat ${file}.cu"
                DEPENDS "${CMAKE_SOURCE_DIR}/src/${file}.cu"
                DEPENDS "${CMAKE_SOURCE_DIR}/src/CudaUtils.h"
        )
        # Because this target depends on the .exe and CMake knows the upper command is responsible
        # for generating this .exe - it will be invoked before running the target. However the command
        # itself is configured to recompile the .exe only if the source file was modified. Clever, huh?
        message("Generating new target: ${file}")
        add_custom_target(
                ${file} ALL
                DEPENDS "${CMAKE_SOURCE_DIR}/build/${file}.exe"
        )
    endforeach ()

else ()
    foreach (file ${files})
        # Get only the filename without the absolute path
        get_filename_component(file ${file} NAME_WE)
        message("Generating new target: ${file}")
        add_executable(${file} src/${file}.cu)
    endforeach ()
endif ()

# To compile this project on Windows, you'll need to have:
#   nVidia GPU
#   CUDA Toolkit
#   Microsoft Visual C++ Build Tools 2015
#
# You'll also need to adjust the variables here and in Compile.bat because most probably you'll have different paths than mine.
#
# Turn on CUSTOM_NVCC_BUILD when compiling inside CLion because it doesn't support Visual Studio compiler natively
# and it's the only Windows compiler that works with NVCC. This is a workaround to make VC++ compiler work in CLion.

cmake_minimum_required(VERSION 3.6)
project(ZZKG_CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CUDA_TOOLKIT_ROOT_DIR "D:/Programy/nVidia/CUDA/v10.0")
set(CUDA_SAMPLES_ROOT_DIR "D:/Programy/nVidia/CUDA/v10.0/Samples")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
set(CUSTOM_NVCC_BUILD true)

include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
include_directories("${CUDA_SAMPLES_ROOT_DIR}/common/inc")

find_package(CUDA QUIET REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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
#                DEPENDS "${CMAKE_SOURCE_DIR}/src/constants/*.h"
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
        cuda_add_executable(
                ${file}
                src/${file}.cu
        )
    endforeach ()
endif ()

# Dummy executable needed for CLion to provide syntax assistance
file (GLOB SOURCES src/**.cpp src/**.cu src/**.h)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_executable(dummy ${SOURCES})

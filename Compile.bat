@echo off

set PROJECT_DIR=%~dp0
set MS_BUILD_TOOLS=C:/Program Files (x86)/Microsoft Visual C++ Build Tools
set CUDA_INCLUDE_DIRS=D:/Programy/nVidia/CUDA/v10.0/Samples/common/inc
set SRC_DIR=src
set BUILD_DIR=build
set VISUAL_COMPILER_VERSION=2015
set COMPUTE_CAPABILITY=52

call "%MS_BUILD_TOOLS%\vcbuildtools.bat" amd64

cd /d %PROJECT_DIR%

::for /f %%f in %1 do (
::    nvcc %SRC_DIR%\%%f --use-local-env --cl-version %VISUAL_COMPILER_VERSION% -I"%CUDA_INCLUDE_DIRS%" -o %OUT_DIR%\%%~nf -Wno-deprecated-gpu-targets
::)
if [%1] == [] (
    echo No source files provided.
    goto :eof
)

if not exist %BUILD_DIR%\NUL mkdir %BUILD_DIR%

:: --cl-version Not needed in CUDA SDK v10.0
:: nvcc %SRC_DIR%\%1 --use-local-env --cl-version %VISUAL_COMPILER_VERSION% -I"%CUDA_INCLUDE_DIRS%" -arch sm_%COMPUTE_CAPABILITY% -o %BUILD_DIR%\%~n1 -Xcompiler "/openmp" -Wno-deprecated-gpu-targets

nvcc %SRC_DIR%\%1 --use-local-env -I"%CUDA_INCLUDE_DIRS%" -arch sm_%COMPUTE_CAPABILITY% -o %BUILD_DIR%\%~n1 -Xcompiler "/openmp" -Wno-deprecated-gpu-targets --expt-extended-lambda
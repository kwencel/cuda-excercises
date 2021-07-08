# CUDA exercises
A large set of exercises in nVidia CUDA programming language.

These examples present various ways you can enhance your experience with CUDA programming, including:
* Custom GPU memory wrappers (see `src/CudaUtils.h`)
* Custom profiling and error handling utilities (see `src/CudaUtils.h`)
* Usage of nVidia's [thrust](https://github.com/NVIDIA/thrust) library
* Usage of C++ templates
* ~~Support for CUDA syntax in CLion~~ (CLion already understands CUDA syntax on its own)

The last exercise `Projekt1` is the biggest one and uses the abovementioned techniques to the fullest.

The exercises' contents and guidelines are located in the `tasks` folder.
They are described only in Polish unfortunately.

# Build prerequisites
    CMake 3.8
    CUDA Toolkit 5.0
    GCC/MSVC compilers compatibile with installed version of CUDA Toolkit
# Build instructions
```
git clone https://github.com/kwencel/cuda-exercises
cd cuda-exercises
cmake .
make
```

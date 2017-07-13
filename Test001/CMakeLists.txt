### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 3.5)
find_package(CUDA QUIET REQUIRED)

# Tell CMake to run moc when necessary:  
set(CMAKE_AUTOMOC ON) 

# As moc files are generated in the binary dir, tell CMake  
# to always look for includes there:  
set(CMAKE_INCLUDE_CURRENT_DIR ON)  

# Pass additional compiler flags (non-nvcc):
#set(EXTRA_CXX_FLAGS "--std=c++11" CACHE STRING "common C++ build flags")  
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  

# Pass options to NVCC
#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_22,code=sm_22
#    )

# Specify include directories
#include_directories(
#    kernels
#    utility
#    )

# Specify library paths
#link_directories(
#    /opt/foobar/lib
#    /opt/joestuff/lib
#    )

#ALL the source files  
#set(SOURCES  
#   src/hellocuda.cu
#)  

#cuda_compile(CUDA_OBJECTS cudakernel.cu)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    hellocuda
    src/hellocuda.cu
#    src/hellocuda.h
#    kernels/hellokernels.cu
#    kernels/hellokernels.h
#    utility/wrapper.cpp
#    utility/wrapper.h
    )

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#    hellocuda
#    -lfoobar
#    -ljoestuff
#    )

#Build the rest of the executable.  
#add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS} ${UI_HEADERS} ${CUDA_OBJECTS})  

#Remember dependencies!  
#target_link_libraries(${EXECUTABLE_NAME} GL ${CUDA_LIBRARIES})  

#Add the Qt stuff.  
#qt5_use_modules(${EXECUTABLE_NAME} Widgets OpenGL)  

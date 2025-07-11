cmake_minimum_required(VERSION 3.19)

project(2D-WENO Fortran C CXX CUDA)

# set(ENV{KOKKOS_DEVICES} "Cuda")
# set(ENV{KOKKOS_ARCH} "Turing75")
# set(CMAKE_CXX_COMPILER /home/hongwei/kokkos-master/bin/nvcc_wrapper)
list(APPEND CMAKE_PREFIX_PATH "/home/hongwei/kokkos-master")

find_package(Kokkos REQUIRED)# CONFIG PATHS "/home/hongwei/kokkos-master/install/lib/cmake/Kokkos")
# find_package(CUDA REQUIRED)

# include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(/home/hongwei/kokkos-master/install/include)
# link_directories(/home/hongwei/kokkos-master/install/lib/libkokkoscore.a)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")   #--extended-lambda")

# MESSAGE( STATUS "KOKKOS_LIBRARY_DIRS = ${KOKKOS_LIBRARY_DIRS}.")

add_definitions(-DKokkos_ENABLE_OPENMP=ON)  #打开openmp并行
add_definitions(-DKokkos_ENABLE_CUDA=ON)  #打开CUDA

file(GLOB FORTRAN_SOURCES "src_f/*.f90")
file(GLOB CPP_SOURCES "src_c/*.cpp" "src_c/*.hpp")
add_executable(2D-WENO.exe
    ${FORTRAN_SOURCES}
    ${CPP_SOURCES}
)

set_target_properties(
    2D-WENO.exe
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)
target_include_directories(2D-WENO.exe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
        $<INSTALL_INTERFACE:mod>
)
target_include_directories(2D-WENO.exe
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(2D-WENO.exe ${PROJECT_LIBS} flcl::flcl)
# target_link_libraries(2D-WENO.exe Kokkos::kokkos)
if (CMAKE_Fortran_COMPILER_ID STREQUAL "XL")
    target_link_options(2D-WENO.exe PRIVATE LINKER:-lxlf90_r)
endif()
if (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set_target_properties(2D-WENO.exe PROPERTIES LINKER_LANGUAGE CUDA)  #  Fortran
endif()
set(RPATHS "$ENV{LD_LIBRARY_PATH}")
set_target_properties(2D-WENO.exe PROPERTIES 
                      BUILD_WITH_INSTALL_RPATH True
                      INSTALL_RPATH "${RPATHS}")
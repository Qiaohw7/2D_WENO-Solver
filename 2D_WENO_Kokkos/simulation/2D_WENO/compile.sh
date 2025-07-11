#!/bin/bash
export KOKKOS_ROOT_DIR=/home/hongwei/kokkos-master/build/core/src
export Fortran_interface=/home/hongwei/2D_WENO_Kokkos/src
rm *.o *.mod *.x 

cp src_f/*.f90 .
cp src_c/*.cpp .
cp src_c/*.hpp .

gfortran -c -std=f2008 flcl-util-strings-f.f90 flcl-types-f.f90 flcl-ndarray-f.f90 flcl-view-f.f90 flcl-dualview-f.f90 flcl-f.f90
gfortran -c -std=f2008 Constant_mod.f90 Controlpara_mod.f90 Output.f90 WENO5-f.f90 
gfortran -c -std=f2008 flcl-util-kokkos-f.f90
g++ -c -fopenmp -I. -I$KOKKOS_ROOT_DIR/include flcl-util-cxx.cpp flcl-cxx.cpp
g++ -c -fopenmp -I. -I$KOKKOS_ROOT_DIR/include WENO5-c.cpp
gfortran -c -g -std=f2008 WENO5_MAIN.f90
gfortran -std=f2008 -o TEST.x flcl-util-strings-f.o flcl-types-f.o flcl-ndarray-f.o flcl-view-f.o flcl-dualview-f.o flcl-f.o  Constant_mod.o Controlpara_mod.o Output.o WENO5-f.f90 flcl-util-kokkos-f.o flcl-util-cxx.o flcl-cxx.o WENO5-c.o WENO5_MAIN.o -L$KOKKOS_ROOT_DIR/lib -lkokkoscore -lstdc++ -fopenmp

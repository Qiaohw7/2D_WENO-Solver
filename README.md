# 2D_WENO-Solver
High-Performance 2D CFD Solver with Kokkos Parallelization on Multi-Core CPUs and GPUs

Due to the high programming complexity of CUDA and CUDA Fortran, this project employs Kokkos' simplified porting approach to migrate a CPU-based Fortran flow solver to GPU. This attempt serves as a reference for researchers working on similar porting efforts.​​

The project consists of two main components:
  1. ​​Kokkos-Fortran Interface​​: Developed to enable seamless Kokkos calls from Fortran.
  2. ​​WENO Flow Solver Implementation​​: Built under the simulation/directory, this 2D weighted essentially non-oscillatory (WENO) solver uses Kokkos' specialized syntax to automatically translate Fortran code into GPU-executable Kokkos kernels.

The compilation process and file directory structure are detailed in the project's CMakeLists.txt. We welcome discussion and collaboration with fellow researchers.

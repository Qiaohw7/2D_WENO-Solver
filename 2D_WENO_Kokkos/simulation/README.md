A few (starting with one) simple usage examples.

Be sure to at least build the libflcl.a target in the root build directory first.
- cd /home/hongwei/kokkos-fortran-interop
- export KOKKOS_ROOT=kokkos-master
- make libflcl.a
- cd ../examples/01-axpy
- make axpy.x
- ./axpy.x

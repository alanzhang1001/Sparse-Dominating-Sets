// One-thread portability shim for correctness validation without OpenMP.
// Formal GCC/OpenMP builds never include this directory.
#pragma once
#ifdef _OPENMP
#error "Do not use the serial shim in an OpenMP build"
#endif
inline int omp_get_thread_num() { return 0; }

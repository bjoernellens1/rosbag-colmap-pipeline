cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(USE_HIP)
  # project(... LANGUAGES ... HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise
  # auto-detects the host GPU and errors on a no-GPU build host.
  project(caspar_library LANGUAGES CXX HIP)

  find_package(hip REQUIRED)
  find_package(hipcub REQUIRED)
else()
  if(NOT DEFINED CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler")
  endif()

  project(caspar_library LANGUAGES CXX CUDA)

  set(CASPAR_MIN_ARCH 75 CACHE INTERNAL "Minimum CUDA arch required by Caspar")

  # SASS for Turing/Ampere/Ada (sm_75–89) + PTX fallback for forward JIT-compatibility. sm_75 minimum required by cooperative_groups::reduce.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CASPAR_MIN_ARCH} 80-real 86-real 89-real)
  endif()

  find_package(CUDAToolkit REQUIRED)
endif()

file(GLOB CUDA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu" "*.cc")
list(FILTER CUDA_SOURCES EXCLUDE REGEX "pybind.*")
file(GLOB CUDA_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cuh" "*.h")
list(FILTER CUDA_HEADERS EXCLUDE REGEX "pybind.*")

add_library(caspar_lib_core STATIC ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(caspar_lib_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_HIP)
  # Mark .cu files as HIP language
  set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_definitions(caspar_lib_core PUBLIC USE_HIP)
  # Link hip targets PRIVATE to avoid propagating HIP compile options to pure-CXX consumers.
  # The static library has the HIP code linked in; pybind consumers only need the includes.
  target_link_libraries(caspar_lib_core PRIVATE hip::host hip::hipcub)
  # LOCAL PATCH (hip-integration branch, 2026-08-05) -- do not lose this on
  # regeneration; see docs/rocm-integration.md Task 4. Everything else in
  # this if(USE_HIP) branch (find_package(hip)/find_package(hipcub),
  # LANGUAGE HIP source properties, HIP_ARCHITECTURES) comes from
  # symforce-rocm's own codegen templates unconditionally -- only the
  # get_target_property(hip::host ...)/ROCM_PATH-fallback lookup below,
  # through the end of this if(USE_HIP) block, is a hand-patch replacing a
  # no-op ${hip_INCLUDE_DIRS} expression. Re-running generate_caspar.py from
  # an unpatched symforce-rocm codegen template will silently drop it.
  #
  # Expose ROCm include directory for consumers that need hipError_t etc.
  # NOTE: ${hip_INCLUDE_DIRS} is not actually set by modern find_package(hip)
  # (which only populates the hip::host imported target's own
  # INTERFACE_INCLUDE_DIRECTORIES, not a legacy <Package>_INCLUDE_DIRS
  # variable), so that expression silently expanded to nothing and left pure
  # C++ consumers of caspar_lib_core (e.g. colmap's bundle_adjustment.cc,
  # which only transitively needs solver.h -- not any HIP kernel -- but
  # solver.h unconditionally #includes cuda_to_hip.h, which under -DUSE_HIP
  # requires <hip/hip_runtime.h>) building with -DUSE_HIP but no HIP include
  # path: "fatal error: hip/hip_runtime.h: No such file or directory".
  # Pull the real include dirs from the hip::host target instead -- but
  # hip::host's own INTERFACE_INCLUDE_DIRECTORIES can ALSO be empty (the
  # HIP-language compiler wrapper knows its own ROCm sysroot implicitly, so
  # find_package(hip) doesn't always need to publish an explicit -I for
  # HIP-language translation units; plain CXX translation units get no such
  # implicit knowledge). Fall back to ROCM_PATH (set by
  # cmake/FindDependencies.cmake's HIP_ENABLED block, visible here as a
  # CACHE variable) if the target property comes back empty.
  get_target_property(_caspar_hip_host_includes hip::host
                       INTERFACE_INCLUDE_DIRECTORIES)
  if(_caspar_hip_host_includes)
    target_include_directories(caspar_lib_core PUBLIC ${_caspar_hip_host_includes})
  elseif(DEFINED ROCM_PATH AND EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
    target_include_directories(caspar_lib_core PUBLIC "${ROCM_PATH}/include")
  else()
    message(WARNING
      "Caspar-HIP: could not determine ROCm include directory from "
      "hip::host's INTERFACE_INCLUDE_DIRECTORIES or ROCM_PATH "
      "('${ROCM_PATH}') -- pure-C++ consumers of caspar_lib_core "
      "(e.g. colmap's bundle_adjustment.cc, via solver.h -> cuda_to_hip.h) "
      "may fail to find <hip/hip_runtime.h>.")
  endif()
  # Use -ffp-contract=fast -fno-math-errno instead of -ffast-math.
  # clang's -ffast-math enables -fassociative-math which can NaN backward passes.
  target_compile_options(caspar_lib_core PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=fast -fno-math-errno>
  )
  set_target_properties(caspar_lib_core PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )
else()
  target_link_libraries(caspar_lib_core PUBLIC CUDA::cudart)
  target_compile_options(caspar_lib_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

set_target_properties(caspar_lib_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)


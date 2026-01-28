add_library(pylittle_core
    src/engine.cpp
    src/memory_manager.cpp
    src/quant.cpp
  src/kv_pager.cpp
)

target_include_directories(pylittle_core PUBLIC include)

# placeholder for dependencies (e.g., threads, CUDA when needed)
find_package(Threads)
if(Threads_FOUND)
  target_link_libraries(pylittle_core PUBLIC Threads::Threads)
endif()

# Optional CUDA support for MemoryManager async/streams
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  target_compile_definitions(pylittle_core PUBLIC PYLITTLE_WITH_CUDA)
  target_link_libraries(pylittle_core PUBLIC CUDA::cudart)
endif()

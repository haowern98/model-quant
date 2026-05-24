add_library(model_surgery_runtime SHARED
    src/model_surgery_runtime.cpp
)

target_compile_features(model_surgery_runtime PRIVATE cxx_std_17)
target_include_directories(model_surgery_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(model_surgery_runtime PRIVATE llama)
target_compile_definitions(model_surgery_runtime PRIVATE MODEL_SURGERY_RUNTIME_BUILD)

if (MSVC)
    target_compile_options(model_surgery_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(model_surgery_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(model_surgery_runtime_probe
    tools/probe.cpp
)
target_link_libraries(model_surgery_runtime_probe PRIVATE model_surgery_runtime)
